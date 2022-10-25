from collections import OrderedDict

import torch
import torch.nn as nn
from .CSPdarknet import CSPdarknet53
from torch.autograd import Function


def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(
        OrderedDict([
            ("conv",
             nn.Conv2d(filter_in,
                       filter_out,
                       kernel_size,
                       stride,
                       pad,
                       bias=False)),
            ("bn", nn.BatchNorm2d(filter_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))


#---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=pool_size,
                         stride=1,
                         padding=pool_size // 2) for pool_size in pool_sizes
        ])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1), 
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   三次卷积块
#---------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], kernel_size=1),
        conv2d(filters_list[0], filters_list[1], kernel_size=3), 
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   五次卷积块
#---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], kernel_size=1), 
        conv2d(filters_list[0], filters_list[1], kernel_size=3),
        conv2d(filters_list[1], filters_list[0], kernel_size=1),
        conv2d(filters_list[0], filters_list[1], kernel_size=3),
        conv2d(filters_list[1], filters_list[0], kernel_size=1),
    )
    return m

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], kernel_size=3), 
        nn.Conv2d(filters_list[0], filters_list[1], kernel_size=1),
    )
    return m

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors, anchors_mask, num_classes, input_shape, backbone_path, pretrained=True, phase='inference'):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成CSPdarknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone = CSPdarknet53(pretrained, backbone_path)

        self.conv_set1 = make_three_conv([512, 1024], 1024)
        self.SPP = SpatialPyramidPooling()
        self.conv_set2 = make_three_conv([512, 1024], 2048)

        self.upsample1 = Upsample(512, 256)
        self.conv_for_P4 = conv2d(512, 256, kernel_size=1, stride=1)
        self.conv_set3 = make_five_conv([256, 512], 512)

        self.upsample2 = Upsample(256, 128)
        self.conv_for_P3 = conv2d(256, 128, kernel_size=1, stride=1)
        self.conv_set4 = make_five_conv([128, 256], 256)
        # 3*(5+num_classes),52,52
        self.yolo_head3 = yolo_head([256, len(anchors_mask[0])*(5+num_classes)], 128)

        self.downsample1 = conv2d(128, 256, kernel_size=3, stride=2)
        self.conv_set5 = make_five_conv([256, 512], 512)
        # 3*(5+num_classes),26,26
        self.yolo_head2 = yolo_head([512, len(anchors_mask[1])*(5+num_classes)], 256)

        self.downsample2 = conv2d(256, 512, kernel_size=3, stride=2)
        self.conv_set6 = make_five_conv([512, 1024], 1024)
        # 3*(5+num_classes),13,13
        self.yolo_head1 = yolo_head([1024, len(anchors_mask[2])*(5+num_classes)], 512)

        self.phase = phase  # 指定是train 还是inference
        # 推测时使用Detect类
        if self.phase == "inference":  #推测模式
            self.decode = Decode(anchors, num_classes, input_shape)

    def forward(self, x):
        # backbone
        x2, x1, x0 = self.backbone(x)

        # 1024,13,13 -> 512,13,13 -> 1024,13,13 -> 512,13,13 -> 2048,13,13
        P5 = self.conv_set1(x0)
        P5 = self.SPP(P5)
        # 2048,13,13 -> 512,13,13 -> 1024,13,13 -> 512,13,13
        P5 = self.conv_set2(P5)

        # 512,13,13 -> 256,13,13 -> 256,26,26
        P5_upsample = self.upsample1(P5)
        # 512,26,26 -> 256,26,26
        P4 = self.conv_for_P4(x1)
        # 256,26,26 + 256,26,26 = 512,26,26
        P4 = torch.cat([P4, P5_upsample], axis=1)
        # 512,26,26 -> 256,26,26 -> 512,26,26 -> 256,26,26 -> 512,26,26 -> 256,26,26
        P4 = self.conv_set3(P4)

        # 256,26,26 -> 128,26,26 -> 128,52,52
        P4_upsample = self.upsample2(P4)
        # 256,52,52 -> 128,52,52
        P3 = self.conv_for_P3(x2)
        # 128,52,52 + 128,52,52 = 256,52,52
        P3 = torch.cat([P3, P4_upsample], axis=1)
        # 256,52,52 -> 128,52,52 -> 256,52,52 -> 128,52,52 -> 256,52,52 -> 128,52,52
        P3 = self.conv_set4(P3)
        
        # 128,52,52 -> 256,26,26
        P3_downsample = self.downsample1(P3)
        # 256,26,26 + 256,26,26 = 512,26,26
        P4 = torch.cat([P3_downsample, P4], axis=1)
        # 512,26,26 -> 256,26,26 -> 512,26,26 -> 256,26,26 -> 512,26,26 -> 256,26,26
        P4 = self.conv_set5(P4)

        # 256,26,26 -> 512,13,13
        P4_downsample = self.downsample2(P4)
        # 512,13,13 + 512,13,13 = 1024,13,13
        P5 = torch.cat([P4_downsample, P5], axis=1)
        # 1024,13,13 -> 512,13,13 -> 1024,13,13 -> 512,13,13 -> 1024,13,13 -> 512,13,13
        P5 = self.conv_set6(P5)

        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,3*(5+num_classes),13,13)
        #---------------------------------------------------#
        out0 = self.yolo_head1(P5)
        
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,3*(5+num_classes),26,26)
        #---------------------------------------------------#
        out1 = self.yolo_head2(P4)

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,3*(5+num_classes),52,52)
        #---------------------------------------------------#
        out2 = self.yolo_head3(P3)

        if self.phase == "inference":  #推测模式
            # 执行Detect类的forward
            return self.decode.forward([out0, out1, out2])
        else: # 学习模式
            return out0, out1, out2


class Decode(Function):
    '''
    相对位置得到对应的绝对位置比如之前的位置是0.5,0.5变为 11.5, 11.5这样的
    '''
    def __init__(self, anchors, num_classes, input_shape, anchors_mask=[[6,7,8],[3,4,5],[0,1,2]]):
        #-----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[142, 110],[192, 243],[459, 401]
        #   26x26的特征层对应的anchor是[36, 75],[76, 55],[72, 146]
        #   52x52的特征层对应的anchor是[12, 16],[19, 36],[40, 28]
        #-----------------------------------------------------------#
        self.anchors_mask   = anchors_mask
        self.anchors        = anchors
        # self.num_anchors    = len(anchors)
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        self.grid_size      = None  # 特征图大小

    def create_grids(self, grid_size, l, cuda=False):
        """
        更新grids信息并生成新的grids参数
        :param grid_size: 特征图大小
        :param device:
        :return:
        """
        self.grid_size = grid_size
        in_w, in_h = self.grid_size

        #-----------------------------------------------#
        #   输入为416x416时
        #   stride_h = stride_w = 32、16、8
        #-----------------------------------------------#
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w

        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        #-----------------------------------------------------#
        #   生成网格，先验框中心，网格左上角
        #-----------------------------------------------------#
        # Calculate offsets for each grid
        # 值为1的维度对应的值不是固定值，后续操作可根据broadcast广播机制自动扩充
        grid_x = torch.arange(in_w).repeat(in_h, 1).view([1, 1, in_h, in_w]).type(FloatTensor)
        grid_y = torch.arange(in_h).repeat(in_w, 1).t().view([1, 1, in_h, in_w]).type(FloatTensor)

        #-------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        #-------------------------------------------------#
        scaled_anchors = FloatTensor([(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors[self.anchors_mask[l]]])
        anchor_w = scaled_anchors[:, 0:1].view((1, len(self.anchors_mask[l]), 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, len(self.anchors_mask[l]), 1, 1))
        
        return grid_x, grid_y, anchor_w, anchor_h

    def forward(self, ps):
        outputs = []
        for l, p in enumerate(ps):
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
            prediction = p.view(batch_size, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

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
                grid_x, grid_y, anchor_w, anchor_h = self.create_grids(grid_size, l, cuda=p.is_cuda) 
            
            #----------------------------------------------------------#
            #   利用预测结果对先验框进行调整
            #   首先调整先验框的中心，从先验框中心向右下角偏移
            #   再调整先验框的宽高。
            #----------------------------------------------------------#
            io = prediction.clone()  # inference output
            io[..., 0] = x + grid_x
            io[..., 1] = y + grid_y
            io[..., 2] = torch.exp(w.data) * anchor_w
            io[..., 3] = torch.exp(h.data) * anchor_h
            io[..., 4:5] = pred_conf
            io[..., 5:]  = pred_cls

            #----------------------------------------------------------#
            #   将输出结果归一化成小数的形式
            #----------------------------------------------------------#
            _scale = torch.Tensor([in_w, in_h, in_w, in_h]).type_as(x)
            io[..., :4] /= _scale
            # view [1, 3, 13, 13, 85] as [1, 507, 85]
            io = io.view(batch_size, -1, self.bbox_attrs)

            outputs.append(io)
    
        return outputs






