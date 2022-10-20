import math

import numpy as np
import torch
import torch.nn as nn
from utils.utils import xywh2xyxy


class YoloLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(YoloLoss, self).__init__()
        #-----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
        #-----------------------------------------------------------#
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask

        self.giou = True
        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2)
        self.cls_ratio = 1 * (num_classes / 80)

        self.ignore_threshold = 0.5
        self.cuda = cuda

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def box_giou(self, b1, b2):
        """
        输入为：
        ----------
        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        返回为：
        -------
        giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        #----------------------------------------------------#
        #   求出预测框左上角右下角
        #----------------------------------------------------#
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        #----------------------------------------------------#
        #   求出真实框左上角右下角
        #----------------------------------------------------#
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        #----------------------------------------------------#
        #   求真实框和预测框所有的iou
        #----------------------------------------------------#
        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / union_area

        #----------------------------------------------------#
        #   找到包裹两个框的最小框的左上角和右下角
        #----------------------------------------------------#
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(enclose_maxes))
        #----------------------------------------------------#
        #   计算对角线距离
        #----------------------------------------------------#
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        giou = iou - (enclose_area - union_area) / enclose_area

        return giou

    def calculate_iou(self, _box_a, _box_b):
        #-----------------------------------------------------------#
        #   将真实框和预测框都转化成左上角右下角的形式
        #-----------------------------------------------------------#
        box_a = xywh2xyxy(_box_a)
        box_b = xywh2xyxy(_box_b)

        #-----------------------------------------------------------#
        #   A为真实框的数量，B为先验框的数量
        #-----------------------------------------------------------#
        A = box_a.size(0)
        B = box_b.size(0)

        #-----------------------------------------------------------#
        #   计算交的面积
        #-----------------------------------------------------------#
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[:, :, 0] * inter[:, :, 1]
        #-----------------------------------------------------------#
        #   计算预测框和真实框各自的面积
        #-----------------------------------------------------------#
        area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter) # [A, B]
        area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter) # [A, B]
        #-----------------------------------------------------------#
        #   求IOU
        #-----------------------------------------------------------#
        union = area_a + area_b - inter
        return inter / union  # [A, B]

    def forward(self, outputs, predictions, targets=None):
        loss_total = 0
        for l, (output, prediction) in enumerate(zip(outputs, predictions)):
            # outputs: [bs, anchor*w_grid*h_grid, xywh + obj + classes]
            # prediction: [bs, anchor, w_grid, h_grid, xywh + obj + classes] 
            # 
            #--------------------------------#
            #   获得图片数量，特征层的高和宽
            #--------------------------------#
            batch_size = prediction.size(0)
            num_anchors_l = prediction.size(1)
            grid_w = prediction.size(2)
            grid_h = prediction.size(3)
            #-----------------------------------------------------------------------#
            #   计算步长
            #   每一个特征点对应原来的图片上多少个像素点
            #   如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
            #   如果特征层为26x26的话，一个特征点就对应原来的图片上的16个像素点
            #   如果特征层为52x52的话，一个特征点就对应原来的图片上的8个像素点
            #   stride_h = stride_w = 32、16、8
            #-----------------------------------------------------------------------#
            stride_h = self.input_shape[0] / grid_h
            stride_w = self.input_shape[1] / grid_w
            #-------------------------------------------------#
            #   此时获得的scaled_anchors大小是相对于特征层的
            #-------------------------------------------------#
            scaled_anchors = np.array([(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors])

            # FloatTensor = torch.cuda.FloatTensor if prediction.is_cuda else torch.FloatTensor

            pred_boxes = output[..., :4].view(batch_size, num_anchors_l, grid_w, grid_h, -1)  # [bs, anchor, w_grid, h_grid, xywh]
            # 换算映射回特征图尺度
            pred_boxes[..., [0, 2]] = pred_boxes[..., [0, 2]] * grid_w
            pred_boxes[..., [1, 3]] = pred_boxes[..., [1, 3]] * grid_h

            pred_conf = output[..., 4].view(batch_size, num_anchors_l, grid_w, grid_h)
            pred_cls = output[..., 5:].view(batch_size, num_anchors_l, grid_w, grid_h, -1)
            #-----------------------------------------------#
            #   获得网络应该有的预测结果
            #-----------------------------------------------#
            y_true, noobj_mask, box_loss_scale = self.build_target(pred_boxes, targets, scaled_anchors[self.anchors_mask[l]], grid_h, grid_w)
            
            if self.cuda:
                y_true = y_true.type_as(pred_conf)
                noobj_mask = noobj_mask.type_as(pred_conf)
                box_loss_scale = box_loss_scale.type_as(pred_conf)

            #--------------------------------------------------------------------------#
            #   box_loss_scale是真实框宽高的乘积，宽高均在0-1之间，因此乘积也在0-1之间。
            #   2-宽高的乘积: 代表真实框越大，比重越小，小框的比重更大。
            #--------------------------------------------------------------------------#
            box_loss_scale = 2 - box_loss_scale

            obj_mask = y_true[..., 4] == 1
            n = torch.sum(obj_mask)
            loss_loc = loss_cls = loss_conf = 0
            if n != 0:
                if self.giou:
                    #---------------------------------------------------------------#
                    #   计算预测结果和真实结果的giou
                    #----------------------------------------------------------------#
                    giou = self.box_giou(pred_boxes, y_true[..., :4]).type_as(pred_conf)
                    loss_loc = torch.mean((1 - giou)[obj_mask])
                else:
                    x = prediction[..., 0]
                    y = prediction[..., 1]
                    h = prediction[..., 2]
                    w = prediction[..., 3]
                    #-----------------------------------------------------------#
                    #   计算中心偏移情况的loss，使用BCELoss效果好一些
                    #-----------------------------------------------------------#
                    loss_x = torch.mean(self.BCELoss(x[obj_mask], y_true[..., 0][obj_mask]) * box_loss_scale[obj_mask])
                    loss_y = torch.mean(self.BCELoss(y[obj_mask], y_true[..., 1][obj_mask]) * box_loss_scale[obj_mask])
                    #-----------------------------------------------------------#
                    #   计算宽高调整值的loss
                    #-----------------------------------------------------------#
                    loss_w = torch.mean(self.MSELoss(w[obj_mask], y_true[..., 2][obj_mask]) * box_loss_scale[obj_mask])
                    loss_h = torch.mean(self.MSELoss(h[obj_mask], y_true[..., 3][obj_mask]) * box_loss_scale[obj_mask])
                    loss_loc =  0.1 * (loss_x + loss_y + loss_h + loss_w)

                # 分类损失
                loss_cls = torch.mean(self.BCELoss(pred_cls[obj_mask], y_true[..., 5:][obj_mask]))

            loss_conf = torch.mean(self.BCELoss(pred_conf, obj_mask.type_as(pred_conf))[noobj_mask.bool() | obj_mask])

            loss = loss_loc * self.box_ratio + loss_cls * self.cls_ratio + loss_conf * self.balance[l] * self.obj_ratio
            # if n != 0:
            #     print(loss_loc * self.box_ratio, loss_cls * self.cls_ratio, loss_conf * self.balance[l] * self.obj_ratio)
            loss_total += loss

        return loss_total

    def build_target(self, pred_boxes, targets, scaled_anchors_l, grid_h, grid_w):
        #-----------------------------------------------------#
        #   计算一共有多少张图片
        #-----------------------------------------------------#
        bs = len(targets)
        #-----------------------------------------------------#
        #   用于选取哪些先验框不包含物体
        #-----------------------------------------------------#
        noobj_mask = torch.ones(bs, len(scaled_anchors_l), grid_h, grid_w, requires_grad=False)
        #-----------------------------------------------------#
        #   让网络更加去关注小目标
        #-----------------------------------------------------#
        box_loss_scale = torch.zeros(bs, len(scaled_anchors_l), grid_h, grid_w, requires_grad=False)
        #-----------------------------------------------------#
        #   batch_size, 3, 13, 13, 5 + num_classes
        #-----------------------------------------------------#
        y_true = torch.zeros(bs, len(scaled_anchors_l), grid_h, grid_w, self.bbox_attrs, requires_grad=False)
        
        for b in range(bs):
            if len(targets[b]) == 0:
                continue
            batch_target = torch.zeros_like(targets[b])
            #-------------------------------------------------------#
            #   计算真实框，并把真实框转换成相对于特征层的大小
            #   gt_box      num_true_box, 4
            #-------------------------------------------------------#
            batch_target[:, [0,2]] = targets[b][:, [0,2]] * grid_w
            batch_target[:, [1,3]] = targets[b][:, [1,3]] * grid_h
            batch_target[:, 4] = targets[b][:, 4]

            FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
            #-------------------------------------------------------#
            #   将真实框转换一个形式
            #   num_true_box, 4
            #-------------------------------------------------------#
            gt_box = FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)).type(FloatTensor), 
                                            batch_target[:, 2:4]), 1))
            #-------------------------------------------------------#
            #   将先验框转换一个形式
            #   3, 4
            #-------------------------------------------------------#
            anchors_shapes = FloatTensor(torch.cat((torch.zeros((len(scaled_anchors_l), 2)).type(FloatTensor), 
                                                    FloatTensor(scaled_anchors_l)), 1))
            #-------------------------------------------------------#
            #   计算交并比
            #   self.calculate_iou(gt_box, anchor_shapes) = [num_true_box, 9]每一个真实框和9个先验框的重合情况
            #   best_ns: [每个真实框最大的重合度max_iou, 每一个真实框最重合的先验框的序号]
            #-------------------------------------------------------#
            best_ns = torch.argmax(self.calculate_iou(gt_box, anchors_shapes), dim=-1)

            for t, best_n in enumerate(best_ns):
                #----------------------------------------#
                #   判断这个先验框是当前特征图的哪一个先验框
                #----------------------------------------#
                k = best_n
                #----------------------------------------#
                #   获得真实框属于哪个网格点
                #----------------------------------------#
                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()
                #----------------------------------------#
                #   取出真实框的种类
                #----------------------------------------#
                c = batch_target[t, 4].long()

                #----------------------------------------#
                #   noobj_mask代表无目标的特征点
                #----------------------------------------#
                noobj_mask[b, k, j, i] = 0
                #----------------------------------------#
                #   tx、ty代表中心调整参数的真实值
                #----------------------------------------#
                if not self.giou:
                    #----------------------------------------#
                    #   tx、ty代表中心调整参数的真实值, 
                    #   相对于cell左上角的偏移量
                    #----------------------------------------#
                    y_true[b, k, j, i, 0] = batch_target[t, 0] - i.float()
                    y_true[b, k, j, i, 1] = batch_target[t, 1] - j.float()
                    y_true[b, k, j, i, 2] = math.log(batch_target[t, 2] / scaled_anchors_l[best_n][0])
                    y_true[b, k, j, i, 3] = math.log(batch_target[t, 3] / scaled_anchors_l[best_n][1])
                    y_true[b, k, j, i, 4] = 1
                    y_true[b, k, j, i, c + 5] = 1
                else:
                    #----------------------------------------#
                    #   tx、ty代表中心调整参数的真实值
                    #----------------------------------------#
                    y_true[b, k, j, i, 0] = batch_target[t, 0]
                    y_true[b, k, j, i, 1] = batch_target[t, 1]
                    y_true[b, k, j, i, 2] = batch_target[t, 2]
                    y_true[b, k, j, i, 3] = batch_target[t, 3]
                    y_true[b, k, j, i, 4] = 1
                    y_true[b, k, j, i, c + 5] = 1

                #----------------------------------------#
                #   用于获得xywh的比例
                #   大目标loss权重小，小目标loss权重大
                #----------------------------------------#
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / grid_w / grid_h

            
            #-------------------------------------------------------#
            #   将预测结果转换一个形式
            #   pred_boxes_for_ignore      num_anchors, 4
            #-------------------------------------------------------#
            pred_boxed_for_ignore = pred_boxes[b].view(-1, 4)
            #-------------------------------------------------------#
            #   计算交并比
            #   anch_ious       num_true_box, num_anchors
            #-------------------------------------------------------#
            anch_ious = self.calculate_iou(batch_target[:, 0:4], pred_boxed_for_ignore)
            #-------------------------------------------------------#
            #   每个先验框对应真实框的最大重合度
            #   anch_ious_max   num_anchors
            #-------------------------------------------------------#
            anch_ious_max, _ = torch.max(anch_ious, dim=0)
            anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])
            noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0

        return y_true, noobj_mask, box_loss_scale


