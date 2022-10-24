import math

import numpy as np
import torch
import torch.nn as nn
from utils.utils import bbox_iou, bbox_wh_iou, wh_iou


def smooth_BCE(
    eps=0.1
):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t)**self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class YoloLoss():
    def __init__(self,
                 anchors,
                 num_classes,
                 input_shape,
                 device,
                 anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 label_smoothing=0.0,
                 cls_pw=1.0,  # cls BCELoss positive_weight
                 obj_pw=1.0,  # obj BCELoss positive_weight
                 focal_loss=False,
                 alpha=0.25,
                 gamma=2,
                 target_threshold=0.2,
                 iou_type='ciou'):
        #-----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[142, 110],[192, 243],[459, 401]
        #   26x26的特征层对应的anchor是[36, 75],[76, 55],[72, 146]
        #   52x52的特征层对应的anchor是[12, 16],[19, 36],[40, 28]
        #-----------------------------------------------------------#
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask

        self.gr = 1.0
        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 5 * (input_shape[0] * input_shape[1]) / (416**2)
        self.cls_ratio = 1 * (num_classes / 80)

        # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=label_smoothing)

        self.device = device

        # Define criteria
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cls_pw], device=self.device),
                                           reduction='mean')
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([obj_pw], device=self.device),
                                           reduction='mean')
        # focal loss
        if focal_loss:
            self.BCEcls = FocalLoss(self.BCEcls, gamma, alpha)
            self.BCEobj = FocalLoss(self.BCEobj, gamma, alpha)

        self.iou_type = iou_type

        self.target_threshold = target_threshold

    def __call__(self, ps, targets=None):
        cls_loss = torch.zeros(1, device=self.device)  # class loss: Tensor(0)
        box_loss = torch.zeros(1, device=self.device)  # box loss: Tensor(0)
        obj_loss = torch.zeros(1, device=self.device)  # object loss: Tensor(0)
        # Losses
        for l, p in enumerate(ps):  # layer index, layer predictions
            # l 代表使用的是第几个有效特征层
            # p: [bs, anchor*(xywh + obj + classes), h_grid, w_grid]
            batch_size = p.size(0)
            na = p.size(1) # 每个格子对应了多少个anchor
            in_h = p.size(2)
            in_w = p.size(3)
            #  batch_size, anchor, h_grid, w_grid, (num_classes+1+4)
            p = p.view(batch_size, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
            
            #-----------------------------------------------------------------------#
            #   计算步长
            #   每一个特征点对应原来的图片上多少个像素点
            #   stride_h = stride_w = 32、16、8
            #-----------------------------------------------------------------------#
            stride_h = self.input_shape[0] / in_h
            stride_w = self.input_shape[1] / in_w
            #-------------------------------------------------#
            #   此时获得的scaled_anchors大小是相对于特征层的
            #-------------------------------------------------#
            anchors = torch.tensor([[a_w / stride_w, a_h / stride_h] for a_w, a_h in self.anchors[self.anchors_mask[l]]])

            t_cls, t_box, indices, anchor = self.targets_match(p, anchors, targets)

            b, a, gj, gi = indices  # image, anchor, gridy, gridx

            n = b.shape[0]  # number of positive samples

            #----------------------------------------------------------#
            #  损失计算
            #----------------------------------------------------------#
            t_obj = torch.zeros_like(p[..., 0], device=self.device)  # target obj (batch_size, anchor, h_grid, w_grid, 1)
            if n:
                # 对应匹配到正样本的预测信息
                pb = p[b, a, gj, gi]  # target-subset of predictions

                # CIoU
                pxy = pb[:, :2].sigmoid()
                pwh = pb[:, 2:4].exp().clamp(max=1e3) * anchor
                p_box = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(p_box, t_box, iou_type=self.iou_type)
                box_loss += (1.0 - iou).mean()  # giou loss

                # Classification
                if self.num_classes > 1:  # cls loss (only if multiple classes)
                    # targets
                    t_label = torch.full_like(pb[:, 5:], self.cn, device=self.device)
                    t_label[range(n), t_cls] = self.cp
                    cls_loss += self.BCEcls(pb[:, 5:], t_label)  # BCE

                # Objectness置信度损失
                iou = iou.detach().clamp(0).type(t_obj.dtype)
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                t_obj[b, a, gj, gi] = iou  # iou ratio

            obj_loss += self.balance[l] * self.BCEobj(pb[..., 4], t_obj)  # obj loss

        loss = sum([self.box_ratio * box_loss, 
                    self.obj_ratio * obj_loss,
                    self.cls_ratio * cls_loss])
        return loss

    def targets_match(self, p, anchors, targets):
        #-----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   p: batch_size, (num_classes+1+4) x num_anchr, grid_h, grid_w
        #   targets: 真实框的标签情况(image_idx,class,x,y,w,h)
        #-----------------------------------------------#
        in_h = p.size(2)
        in_w = p.size(3)

        # normalized to gridspace gain
        gain = torch.ones(6, device=targets.device).long()
        # gain[2:6] = torch.tensor(p.shape)[[3, 2, 3, 2]]  # xyxy gain
        gain[2:6] = torch.tensor([in_w, in_h, in_w, in_h])  # xyxy gain

        num_a = anchors.shape[0]  # 特征层上anchor的数量
        num_t = targets.shape[0]  # gt的数量
        # [3] -> [3, 1] -> [3, num_t]
        # anchor tensor, same as .repeat_interleave(nt)
        at = torch.arange(num_a).view(num_a, 1).repeat(1, num_t)

        # Match targets to anchors
        a = []
        t = targets * gain
        offsets = 0
        if num_t:  # 如果存在target
            # 通过计算anchor模板与所有target的wh_iou来匹配正样本
            # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
            # j: [3, num_t] , iou_t = 0.20
            j = wh_iou(anchors, t[:, 4:6]) > self.target_threshold
            # t.repeat(na, 1, 1): [num_t, 6] -> [3, num_t, 6]
            # 获取正样本对应的anchor模板与target信息
            a, t = at[j], t.repeat(num_a, 1, 1)[j]  # filter

        # Define
        # long等于to(torch.int64), 数值向下取整
        b, c = t[:, :2].long().T  # image_idx, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()  # 匹配targets所在的grid cell左上角坐标
        gi, gj = gij.T  # grid xy indices

        # Append
        # gain[3]: grid_h, gain[2]: grid_w
        # image_idx, anchor_idx, grid indices(y, x)
        indices = (b, a, gj.clamp_(0, in_w - 1), gi.clamp_(0, in_h - 1))
        # gt box相对grid cell左上角坐标的x,y偏移量以及w,h
        t_box = torch.cat((gxy - gij, gwh), 1)
        anchor = anchors[a]  # anchors的w、h
        t_cls = c
        if c.shape[0]:  # if any targets
            # 目标的标签数值不能大于给定的目标类别数
            assert c.max() < self.num_classes, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g.' % (
                                                self.num_classes, self.num_classes - 1, c.max())
        return t_cls, t_box, indices, anchor