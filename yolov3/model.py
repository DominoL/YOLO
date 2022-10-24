from collections import OrderedDict

import torch
import torch.nn as nn
from yolov3.darknet import darknet53


class YoloBody(nn.Module):
    def __init__(self, num_classes, input_shape, anchors, anchors_mask=[[6,7,8], [3,4,5], [0,1,2]], pretrained=True):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.pretrained = pretrained
        self.backbone = darknet53(self.pretrained)

        #---------------------------------------------------#
        #   out_filters : [64, 128, 256, 512, 1024]
        #---------------------------------------------------#
        out_filters = self.backbone.layers_out_filters

        #------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75（3*(20+4+1)）
        #------------------------------------------------------------------------#
        self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))
        self.yolo_head0 = YOlOHead(anchors[anchors_mask[0]], num_classes, input_shape)
        
        self.last_layer1_conv = conv2d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))
        self.yolo_head1 = YOlOHead(anchors[anchors_mask[1]], num_classes, input_shape)

        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))
        self.yolo_head2 = YOlOHead(anchors[anchors_mask[2]], num_classes, input_shape)

    def forward(self, x):
        # yolo_out收集每个yolo_head层的输出
        yolo_out = []

        #---------------------------------------------------#   
        #   获得三个有效特征层，他们的shape分别是：
        #   256*52*52；512*26*26；1024*13*13
        #---------------------------------------------------#
        x2, x1, x0 = self.backbone(x)

        #---------------------------------------------------#
        #   第一个特征层
        #   out0 = (batch_size,255,13,13)
        #---------------------------------------------------#
        # 1024,13,13 -> 512,13,13 -> 1024,13,13 -> 512,13,13 -> 1024,13,13 -> 512,13,13
        out0_branch = self.last_layer0[:5](x0)
        # 512,13,13 -> 1024,13,13 -> num_classes*(20+4+1),13,13
        out0 = self.last_layer0[5:](out0_branch)
        yolo_out.append(self.yolo_head0(out0))

        # 特征上采样
        # 512,13,13 -> 256,13,13 -> 256,26,26
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        # 特征融合
        # 256,26,26 + 512,26,26 -> 768,26,26
        x1_in = torch.cat([x1_in, x1], 1)

        #---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        #---------------------------------------------------#
        # 768,26,26-> 256,26,26 -> 512,26,26 -> 256,26,26 -> 512,26,26 -> 256,26,26
        out1_branch = self.last_layer1[:5](x1_in)
        # 256,26,26 -> 512,26,26 -> num_classes*(20+4+1),26,26
        out1 = self.last_layer1[5:](out1_branch)
        yolo_out.append(self.yolo_head1(out1))

        # 特征上采样
        # 256,26,26 -> 128,26,26 -> 128,52,52
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)
        # 特征融合
        # 256,52,52 + 128,52,52 -> 384,52,52
        x2_in = torch.cat([x2_in, x2], 1)

        #---------------------------------------------------#
        #   第三个特征层
        #   out2 = (batch_size,255,52,52)
        #---------------------------------------------------#
        # 384,52,52 -> 128,52,52 -> 256,52,52 -> 128,52,52 -> 256,52,52
        # -> 128,52,52 -> 256,52,52 -> num_classes*(20+4+1),52,52
        out2 = self.last_layer2(x2_in)
        yolo_out.append(self.yolo_head2(out2))
        io, p = zip(*yolo_out)  # inference output, training output
        # io = torch.cat(io, 1)  # cat yolo outputs
        return io, p


def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filter, out_filter):
    m = nn.Sequential(
        conv2d(in_filter, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return m


class YOlOHead(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, letterbox_image=True):
        super(YOlOHead, self).__init__()
        #-----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
        #-----------------------------------------------------------#
        self.anchors        = anchors
        self.num_anchors    = len(anchors)
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        self.grid_size      = None  # 特征图大小
        
        self.letterbox_image = letterbox_image

    def create_grids(self, grid_size, cuda=False):
        """
        更新grids信息并生成新的grids参数
        :param grid_size: 特征图大小
        :param device:
        :return:
        """
        self.grid_size = grid_size
        in_w, in_h = self.grid_size

        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        #-----------------------------------------------------#
        #   生成网格，先验框中心，网格左上角
        #-----------------------------------------------------#
        # Calculate offsets for each grid
        # 值为1的维度对应的值不是固定值，后续操作可根据broadcast广播机制自动扩充
        self.grid_x = torch.arange(in_w).repeat(in_h, 1).view([1, 1, in_h, in_w]).type(FloatTensor)
        self.grid_y = torch.arange(in_h).repeat(in_w, 1).t().view([1, 1, in_h, in_w]).type(FloatTensor)

        #-------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        #-------------------------------------------------#
        self.scaled_anchors = FloatTensor([(a_w / self.stride_w, a_h / self.stride_h) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, p):
        #-----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   batch_size, (num_classes+1+4) x 3, 13, 13
        #   batch_size, (num_classes+1+4) x 3, 26, 26
        #   batch_size, (num_classes+1+4) x 3, 52, 52
        #-----------------------------------------------#
        batch_size = p.size(0)
        in_h = p.size(2)
        in_w = p.size(3)
        grid_size = (in_w, in_h)

        #-----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   batch_size, 3, 13, 13, (num_classes+1+4)
        #   batch_size, 3, 26, 26, (num_classes+1+4)
        #   batch_size, 3, 52, 52, (num_classes+1+4)
        #-----------------------------------------------#
        prediction = p.view(batch_size, self.num_anchors, self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        #-----------------------------------------------#
        #   输入为416x416时
        #   stride_h = stride_w = 32、16、8
        #-----------------------------------------------#
        self.stride_h = self.input_shape[0] / in_h
        self.stride_w = self.input_shape[1] / in_w

        # Calculate offsets for each grid
        # build xy offsets 构建每个cell处的anchor的xy偏移量(在feature map上的)
        #-----------------------------------------------#
        #   先验框的中心位置的调整参数
        #-----------------------------------------------#
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        #-----------------------------------------------#
        #   先验框的宽高调整参数
        #-----------------------------------------------#
        w = prediction[..., 2]
        h = prediction[..., 3]
        #-----------------------------------------------#
        #   获得置信度，是否有物体
        #-----------------------------------------------#
        pred_conf = torch.sigmoid(prediction[..., 4:5])
        #-----------------------------------------------#
        #   种类置信度
        #-----------------------------------------------#
        pred_cls = torch.sigmoid(prediction[..., 5:])

        

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            #相对位置得到对应的绝对位置比如之前的位置是0.5,0.5变为 11.5，11.5这样的
            self.create_grids(grid_size, cuda=p.is_cuda) 
        
        #----------------------------------------------------------#
        #   利用预测结果对先验框进行调整
        #   首先调整先验框的中心，从先验框中心向右下角偏移
        #   再调整先验框的宽高。
        #----------------------------------------------------------#
        io = prediction.clone()  # inference output
        io[..., 0] = x + self.grid_x
        io[..., 1] = y + self.grid_y
        io[..., 2] = torch.exp(w.data) * self.anchor_w
        io[..., 3] = torch.exp(h.data) * self.anchor_h
        io[..., 4:5] = pred_conf
        io[..., 5:]  = pred_cls

        # # 换算映射回原图尺度
        # scale = torch.Tensor([self.stride_w, self.stride_h, self.stride_w, self.stride_h]).type_as(p)
        #----------------------------------------------------------#
        #   将输出结果归一化成小数的形式
        #----------------------------------------------------------#
        _scale = torch.Tensor([in_w, in_h, in_w, in_h]).type_as(x)
        io[..., :4] /= _scale
        # view [1, 3, 13, 13, 85] as [1, 507, 85]
        io = io.view(batch_size, -1, self.bbox_attrs)

        #----------------------------------------------------------#
        #   输出在特征图上的结果
        #----------------------------------------------------------#
        # [bs, anchor, grid, grid, xywh + obj + classes] 
        prediction = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1), w.unsqueeze(-1), h.unsqueeze(-1), pred_conf, pred_cls), -1)

        return io, prediction  
