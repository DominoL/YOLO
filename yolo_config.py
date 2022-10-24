from utils.utils import get_anchors, get_classes


class YoloConfig():
    def __init__(self, version):
        # 训练相关
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        self.input_shape        = [416, 416]
        self.version            = version
        if version == 'v3':
            self.backbone_path      = '.\\weights\\v3\\darknet53_backbone_weights.pth'
            self.model_path         = '.\\weights\\v3\\yolo3_weights.pth'
            self.anchors_path       = '.\\data\\yolov3_anchors.txt'
        if version == 'v4':
            self.backbone_path      = '.\\weights\\v4\\CSPdarknet53_backbone_weights.pth'
            self.model_path         = '.\\weights\\v4\\yolo4_weights.pth'
            self.anchors_path       = '.\\data\\yolov4_anchors.txt'

        self.anchors_mask       = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.classes_path       = '.\\data\\voc_classes.names'
        self.anchors, self.num_anchors     = get_anchors(self.anchors_path)
        self.class_names, self.num_classes = get_classes(self.classes_path)
        #------------------------------------------------------------------#
        #   save_dir        权值与日志文件保存的文件夹
        #------------------------------------------------------------------#
        self.save_logs = ".\\logs"
        self.save_weights = ".\\weights"

        #------------------------------------------------------------------#
        #   冻结阶段训练参数
        #   此时模型的主干被冻结了，特征提取网络不发生改变
        #   占用的显存较小，仅对网络进行微调
        #   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
        #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
        #                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率。
        #                       （断点续练时使用）
        #   Freeze_Epoch        模型冻结训练的Freeze_Epoch
        #                       (当Freeze_Train=False时失效)
        #   Freeze_batch_size   模型冻结训练的batch_size
        #                       (当Freeze_Train=False时失效)
        #------------------------------------------------------------------#
        self.Init_Epoch          = 0
        self.Freeze_Epoch        = 50
        self.Freeze_batch_size   = 8
        #------------------------------------------------------------------#
        #   解冻阶段训练参数
        #   此时模型的主干不被冻结了，特征提取网络会发生改变
        #   占用的显存较大，网络所有的参数都会发生改变
        #   UnFreeze_Epoch          模型总共训练的epoch
        #                           SGD需要更长的时间收敛，因此设置较大的UnFreeze_Epoch
        #                           Adam可以使用相对较小的UnFreeze_Epoch
        #   Unfreeze_batch_size     模型在解冻后的batch_size
        #------------------------------------------------------------------#
        self.UnFreeze_Epoch      = 300
        self.Unfreeze_batch_size = 4

        #------------------------------------------------------------------#
        #   其它训练参数：学习率、优化器、学习率下降有关
        #------------------------------------------------------------------#
        #------------------------------------------------------------------#
        #   Init_lr         模型的最大学习率
        #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
        #------------------------------------------------------------------#
        
        self.Init_lr             = 1e-2
        self.Min_lr              = self.Init_lr * 0.01

        #------------------------------------------------------------------#
        #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
        #                   当使用Adam优化器时建议设置  Init_lr=1e-3
        #                   当使用SGD优化器时建议设置   Init_lr=1e-2
        #   momentum        优化器内部使用到的momentum参数
        #   weight_decay    权值衰减，可防止过拟合
        #                   adam会导致weight_decay错误，使用adam时建议设置为0。
        #------------------------------------------------------------------#
        self.optimizer_type      = "sgd"
        self.momentum            = 0.937
        self.weight_decay        = 5e-4

        self.nbs = 64
        self.lr_limit_max    = 1e-3 if self.optimizer_type == 'adam' else 5e-2
        self.lr_limit_min    = 3e-4 if self.optimizer_type == 'adam' else 5e-4

        #------------------------------------------------------------------#
        #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
        #------------------------------------------------------------------#
        self.lr_decay_type       = "cos"
        #------------------------------------------------------------------#
        #   save_period     多少个epoch保存一次权值
        #------------------------------------------------------------------#
        self.save_period         = 10
        #------------------------------------------------------------------#
        #   eval_flag       是否在训练时进行评估，评估对象为验证集
        #                   安装pycocotools库后，评估体验更佳。
        #   eval_period     代表多少个epoch评估一次，不建议频繁的评估
        #                   评估需要消耗较多的时间，频繁评估会导致训练非常慢
        #   此处获得的mAP会与get_map.py获得的会有所不同，原因有二：
        #   （一）此处获得的mAP为验证集的mAP。
        #   （二）此处设置评估参数较为保守，目的是加快评估速度。
        #------------------------------------------------------------------#
        self.eval_flag           = True
        self.eval_period         = 30


        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        self.conf_thres          = 0.05
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        self.nms_thres           = 0.5

        self.iou_threshold       = 0.5

        self.max_boxes           = 100
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        self.letterbox_image = False
