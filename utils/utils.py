import math
import random
from functools import partial

import cv2
import matplotlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
matplotlib.rc('font', **{'size': 11})

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Reduce randomness (may be slower on Tesla GPUs) # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        cudnn.deterministic = False
        cudnn.benchmark = True

def preprocess_input(image):
    image /= 255.0
    return image

def get_classes(classes_path):
    """
    获得类别名
    """
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#   获得先验框
#---------------------------------------------------#
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)

def resize_image(image, size, letterbox_image):
    iw, ih = image.size
    w, h = size
    #加边框缩放，避免失真
    if letterbox_image:
        scale = min(w/iw, h/ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        #双三次插值算子
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[..., 0] = x[..., 0] + (x[..., 2] - x[..., 0]) / 2 # x center
    y[..., 1] = x[..., 1] + (x[..., 3] - x[..., 1]) / 2 # y center
    y[..., 2] = x[..., 2] - x[..., 0]   # width
    y[..., 3] = x[..., 3] - x[..., 1]   # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def bbox_wh_iou(wh1, wh2):
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area

def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    # iou = inter / (area1 + area2 - inter)
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)

def bbox_iou(p_box, t_box, iou_type=None, x1y1x2y2=False):
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, anchor_num, feat_w, feat_h, 4), xywh
    b2: tensor, shape=(batch, anchor_num, feat_w, feat_h, 4), xywh
    返回为：
    -------
    iou: tensor, shape=(batch, anchor_num, feat_w, feat_h, 1)
    """
    assert iou_type in [None, 'giou', 'diou', 'ciou', 'siou'], 'iou_type is None/giou/diou/ciou/siou'

    if x1y1x2y2:
        b1_x1, b1_y1 = p_box[..., 0], p_box[..., 1]
        b1_x2, b1_y2 = p_box[..., 2], p_box[..., 3]
        b2_x1, b2_y1 = t_box[..., 0], t_box[..., 1]
        b2_x2, b2_y2 = t_box[..., 2], t_box[..., 3]
    else:
        #----------------------------------------------------#
        #   求出预测框左上角右下角
        #----------------------------------------------------#
        box1 = xywh2xyxy(p_box)
        b1_x1, b1_y1 = box1[..., 0], box1[..., 1]
        b1_x2, b1_y2 = box1[..., 2], box1[..., 3]
        #----------------------------------------------------#
        #   求出真实框左上角右下角
        #----------------------------------------------------#
        box2 = xywh2xyxy(t_box)
        b2_x1, b2_y1 = box2[..., 0], box2[..., 1]
        b2_x2, b2_y2 = box2[..., 2], box2[..., 3]

    #----------------------------------------------------#
    #   求真实框和预测框所有的iou
    #----------------------------------------------------#
    # Intersection area
    intersect_w = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0)
    intersect_h = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    intersect_area = intersect_w * intersect_h

    # Union Area
    # 预测框的宽高
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    # 真实框的宽高
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    union_area = (w1 * h1 + 1e-16) + w2 * h2 - intersect_area

    iou = intersect_area / union_area  # iou

    if iou_type is not None:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

        if iou_type == 'giou':  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            # 最小外接矩形的面积
            c_area = cw * ch + 1e-16  # convex area
            giou = iou - (c_area - union_area) / c_area  # GIOU
            return giou
        else:
            # 最小外接矩形对角线的平方
            c2 = cw**2 + ch**2 + 1e-16
            # 中心点距离的平方
            bw = torch.abs((b2_x1 + b2_x2) / 2 - (b1_x1 + b1_x2) / 2)
            bh = torch.abs((b2_y1 + b2_y2) / 2 - (b1_y1 + b1_y2) / 2)
            rho2 = bw**2 + bh**2

            if iou_type == 'diou':  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                diou = iou - rho2 / c2  # DIoU
                return diou

            elif iou_type == 'ciou':  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                alpha = v / (1 - iou + v)
                ciou = iou - (rho2 / c2 + alpha * v)
                return ciou

            elif iou_type == 'siou':  # https://arxiv.org/pdf/2205.12740.pdf
                #----------------------------------------------------#
                #   Angle cost
                #----------------------------------------------------#
                # 计算中心距离
                sigma = torch.pow(rho2, 0.5)
                # 求h和w方向上的sin比值
                sin_alpha_1 = bw / sigma
                sin_alpha_2 = bh / sigma
                #----------------------------------------------------#
                #   求门限，二分之根号二，0.707
                #   如果门限大于0.707，代表某个方向的角度大于45°
                #   此时取另一个方向的角度
                #----------------------------------------------------#
                threshold = torch.pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold,
                                        sin_alpha_2, sin_alpha_1)
                # angle_cost = 1 - 2 * torch.pow( torch.sin(torch.arcsin(sin_alpha) - np.pi/4), 2)
                angle_cost = torch.cos(
                    torch.arcsin(sin_alpha) * 2 - np.pi / 2)

                #----------------------------------------------------#
                #   Distance cost
                #----------------------------------------------------#
                #  求中心与外接矩形高宽的比值
                rho_x = torch.pow(bw / cw, 2)
                rho_y = torch.pow(bh / ch, 2)
                gamma = 2 - angle_cost
                distance_cost = 2 - torch.exp(
                    -1 * gamma * rho_x) - torch.exp(-1 * gamma * rho_y)

                #----------------------------------------------------#
                #   Shape cost
                #   真实框与预测框的宽高差异与最大值的比值
                #   差异越小，costshape_cost越小
                #----------------------------------------------------#
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                siou = iou - 0.5 * (distance_cost + shape_cost)
                return siou
    else:
        return iou


def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, gain=init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1, no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    """
    设置学习率
    """
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_optimizer_lr(optimizer):
    """
    获得学习率
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']