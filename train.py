import os
import time

import numpy as np
import torch
import torch.optim as optim
from callbacks import EvalCallback, History
from config import Config
from dataloader import DataTransform, YoloDataset, yolo_dataset_collate
from nets.losses import YoloLoss
from nets.yolo import Yolo3
from torch import distributed as dist
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils.utils import (get_lr_scheduler, get_optimizer_lr, set_optimizer_lr,
                         weights_init)
from utils.utils_bbox import non_max_suppression

cfg = Config()

class Trainer():
    def __init__(self, gpu, fp16, pretrained, distributed=False, Freeze_Train=True):
        # 训练相关
        self.scaler    = None
        self.optimizer = None
        self.scheduler = None
        self.gpu       = gpu
        self.fp16      = fp16
        self.distributed = distributed    # 多卡平行运行
        self.pretrained  = pretrained
        self.model_path  = cfg.model_path

        self.num_classes = cfg.num_classes
        self.confidence = cfg.conf_thres
        self.nms_iou = cfg.nms_thres

        #------------------------------------------------------------------#
        #   Freeze_Train    是否进行冻结训练
        #                   默认先冻结主干训练后解冻训练。
        #------------------------------------------------------------------#
        self.Freeze_Train        = Freeze_Train

        #------------------------------------------------------#
        #   主干特征提取网络特征通用，冻结训练可以加快训练速度
        #   也可以在训练初期防止权值被破坏。
        #   Init_Epoch为起始世代
        #   Freeze_Epoch为冻结训练的世代
        #   UnFreeze_Epoch总训练世代
        #   提示OOM或者显存不足请调小Batch_size
        #------------------------------------------------------#
        self.UnFreeze_flag = False
        
        #-------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = cfg.Freeze_batch_size if self.Freeze_Train else cfg.Unfreeze_batch_size
        self.batch_size = batch_size
        Init_lr_fit = min(max(batch_size / cfg.nbs * cfg.Init_lr, cfg.lr_limit_min), cfg.lr_limit_max)

        # setting
        self.train_data_loader = self.get_loader("train", batch_size, shuffle=True, num_workers=4)
        self.val_data_loader = self.get_loader("val", batch_size, shuffle=False, num_workers=4)
        self.set_device()
        self.set_model()
        self.scaler = torch.cuda.amp.GradScaler() if self.fp16 else None
        self.set_optimizer(Init_lr_fit)
        self.lr_scheduler = self.get_scheduler(batch_size)
        self.loss = YoloLoss(cfg.anchors, cfg.num_classes, cfg.input_shape, self.device, cfg.anchors_mask)
        self.history = History(cfg.save_logs, self.model, cfg.input_shape)
        self.eval_callback = EvalCallback(cfg.save_logs, cfg.class_names, cfg.input_shape, cfg.iou_threshold)


    def train(self):
        #---------------------------------------#
        #   开始模型训练
        #---------------------------------------#
        for epoch in range(cfg.Init_Epoch, cfg.UnFreeze_Epoch):
            #---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            #---------------------------------------#
            if epoch >= cfg.Freeze_Epoch and not self.UnFreeze_flag and self.Freeze_Train:
                batch_size = cfg.Unfreeze_batch_size
                self.batch_size = batch_size
                self.train_data_loader = self.get_loader("train", batch_size, shuffle=True, num_workers=4)
                self.val_data_loader = self.get_loader("val", batch_size, shuffle=False, num_workers=4)

                for param in self.model.backbone.parameters():
                    param.requires_grad = True

                #---------------------------------------#
                #   获得学习率下降的函数
                #---------------------------------------#
                self.lr_scheduler = self.get_scheduler(batch_size)

                self.UnFreeze_flag = True  # 已解冻
            if self.distributed:
                self.sampler.set_epoch(epoch)
            
            set_optimizer_lr(self.optimizer, self.lr_scheduler, epoch)
            print(f'Epoch {epoch + 1}/{cfg.UnFreeze_Epoch}:')

            eval_flag = True if epoch % cfg.eval_period == 0 and cfg.eval_flag else False

            train_loss = self.train_one_epoch()
            val_loss = self.val_one_epoch(eval_flag)
            if self.local_rank == 0:
                self.history.append_loss(epoch + 1, train_loss, val_loss)

                print('Epoch:'+ str(epoch + 1) + '/' + str(cfg.UnFreeze_Epoch))
                print('Total Loss: %.3f || Val Loss: %.3f ' % (train_loss, val_loss))
                
                if eval_flag:
                    evaluation_metrics = self.eval_callback.get_metrics()
                    print(' || '.join([k+': %.3f' % v for k, v in evaluation_metrics.items()]))

                    self.eval_callback.zero()
                    self.history.append_metrics(epoch + 1, evaluation_metrics)
                
                #-----------------------------------------------#
                #   保存权值
                #-----------------------------------------------#
                if (epoch + 1) % cfg.save_period == 0 or epoch + 1 == cfg.UnFreeze_Epoch:
                    self.save_model(os.path.join(cfg.save_logs, "weights", "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, train_loss, val_loss)))

                if len(self.loss_history.val_loss) <= 1 or (val_loss) <= min(self.loss_history.val_loss):
                    print('Save best model to best_epoch_weights.pth')
                    self.save_model(os.path.join(cfg.save_weights, "best_epoch_weights.pth"))
                    


            if self.distributed:
                dist.barrier()

    def set_optimizer(self, Init_lr_fit):
        #---------------------------------------#
        #   根据optimizer_type选择优化器
        #---------------------------------------#
        pg0, pg1, pg2 = [], [], []  
        for k, v in self.model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   
        self.optimizer = {
            'adam' : optim.Adam(pg0, Init_lr_fit, betas=(cfg.momentum, 0.999)),
            'sgd' : optim.SGD(pg0, Init_lr_fit, momentum=cfg.momentum, nesterov=True)
        }[cfg.optimizer_type]
        self.optimizer.add_param_group({"params": pg1, "weight_decay": cfg.weight_decay})
        self.optimizer.add_param_group({"params": pg2})

    def get_scheduler(self, batch_size):
        Init_lr_fit = min(max(batch_size / 64 * cfg.Init_lr, cfg.lr_limit_min), cfg.lr_limit_max)
        Min_lr_fit = min(max(batch_size / 64 * cfg.Min_lr, cfg.lr_limit_min * 1e-2), cfg.lr_limit_max * 1e-2)
        return get_lr_scheduler(cfg.lr_decay_type, Init_lr_fit, Min_lr_fit, cfg.UnFreeze_Epoch)

    def get_loader(self, phase, batch_size, shuffle, num_workers):
        #---------------------------#
        #   读取数据集对应的txt
        #---------------------------#
        annotation_path = ".\\data\\voc_{}_data.txt".format(phase)
        
        with open(annotation_path) as f:
            annotation_lines = f.readlines()
            if phase == 'train':
                self.num_train = len(annotation_lines)

        dataset = YoloDataset(annotation_lines, cfg.input_shape, transform=DataTransform(), phase=phase)
        if self.distributed:
            ngpus_per_node  = torch.cuda.device_count()
            self.sampler = DistributedSampler(dataset, shuffle=shuffle,)
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            self.sampler = None
            shuffle = True

        data_loader = DataLoader(dataset, 
                                 batch_size=batch_size, 
                                 num_workers=num_workers, 
                                 shuffle=shuffle,
                                 pin_memory=True,
                                 drop_last=True, 
                                 collate_fn=yolo_dataset_collate, 
                                 sampler=self.sampler)

        return data_loader

    def set_device(self):
        #------------------------------------------------------#
        #   设置用到的显卡
        #------------------------------------------------------#
        if self.distributed:
            ngpus_per_node  = torch.cuda.device_count()
            dist.init_process_group(backend="nccl")
            self.local_rank  = int(os.environ["LOCAL_RANK"])
            self.rank        = int(os.environ["RANK"])
            self.device      = torch.device("cuda", self.local_rank)
            if self.local_rank == 0:
                print(f"[{os.getpid()}] (rank = {self.rank}, local_rank = {self.local_rank}) training...")
                print("Gpu Device Count : ", ngpus_per_node)
        else:
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.cuda = torch.cuda.is_available()
            if self.gpu < 0:
                self.cuda = False
            if self.cuda:
                self.device = torch.device("cuda", self.gpu)
            else:
                self.device = torch.device("cpu")
            print("Using {} device training.".format(self.device.type))

            self.local_rank = 0

    def set_model(self):
        #------------------------------------------------------#
        #   创建yolo模型
        #------------------------------------------------------#
        self.model = Yolo3(cfg.num_classes, cfg.input_shape, cfg.anchors, cfg.anchors_mask, self.pretrained, phase='train')
        if not self.pretrained:
            weights_init(self.model)
        else:
            if self.model_path != '':
                #------------------------------------------------------#
                #   权值文件
                #------------------------------------------------------#
                if self.local_rank == 0:
                    print('Load weights {}.'.format(self.model_path))
                
                #------------------------------------------------------#
                #   根据预训练权重的Key和模型的Key进行加载
                #------------------------------------------------------#
                model_dict = self.model.state_dict()
                pretrained_dict = torch.load(self.model_path, map_location=self.device)
                load_key, no_load_key, temp_dict = [], [], {}
                for k, v in pretrained_dict.items():
                    if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                        temp_dict[k] = v
                        load_key.append(k)
                    else:
                        no_load_key.append(k)
                model_dict.update(temp_dict)
                self.model.load_state_dict(model_dict)
                #------------------------------------------------------#
                #   显示没有匹配上的Key
                #------------------------------------------------------#
                if self.local_rank == 0:
                    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
                    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
                    print("\n\033[1;33;44m温馨提示, head部分没有载入是正常现象, Backbone部分没有载入是错误的。\033[0m")
            else:
                print("无保存模型, 将从头开始训练!")

        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        if self.Freeze_Train:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

        self.model.to(self.device)

    def set_epochs(self):
        #---------------------------------------------------------#
        #   总训练世代指的是遍历全部数据的总次数
        #   总训练步长指的是梯度下降的总次数 
        #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
        #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
        #----------------------------------------------------------#
        wanted_step = 5e4 if cfg.optimizer_type == "sgd" else 1.5e4
        total_step  = self.num_train // cfg.Unfreeze_batch_size * cfg.UnFreeze_Epoch
        if total_step <= wanted_step:
            if self.num_train // cfg.Unfreeze_batch_size == 0:
                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
            wanted_epoch = wanted_step // (self.num_train // cfg.Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时, 建议将训练总步长设置到%d以上。\033[0m"%(cfg.optimizer_type, wanted_step))
            print(
                "\033[1;33;44m[Warning] 本次运行的总训练数据量为%d, Unfreeze_batch_size为%d, 共训练%d个Epoch, 计算出总训练步长为%d。\033[0m"
                % (self.num_train, cfg.Unfreeze_batch_size, cfg.UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d, 小于建议总步长%d, 建议设置总世代为%d。\033[0m" 
                  % (total_step, wanted_step, wanted_epoch))
        return wanted_epoch

    def train_one_epoch(self):
        t0 = time.time()
        print('Start Train')
        epoch_step = len(self.train_data_loader)

        # 处理进度条只在第一个进程中显示
        if self.local_rank == 0:
            pbar = tqdm(self.train_data_loader, total=epoch_step, postfix=dict, mininterval=0.3, ascii=True, desc="train")
        else:
            pbar = self.train_data_loader

        loss = 0
        self.model.train()
        for batch_i, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            #----------------------#
            #   清零梯度
            #----------------------#
            self.optimizer.zero_grad()
            # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
            with amp.autocast(enabled=self.scaler is not None):
                #----------------------#
                #   前向传播
                #----------------------#
                ps = self.model(images)
                #----------------------#
                #   计算损失
                #----------------------#
                loss_value = self.loss(ps, targets)

            #----------------------#
            #   反向传播
            #----------------------#
            if self.scaler is not None:
                self.scaler.scale(loss_value).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_value.backward()
                self.optimizer.step()

            loss += loss_value.item()
            time_span = time.time() - t0
            if self.local_rank == 0:
                pbar.set_postfix(**{'time(s)': time_span,
                                    'loss'   : loss / (batch_i + 1), 
                                    'lr'     : get_optimizer_lr(self.optimizer)})

        # print('Finish Train')
            return loss / epoch_step
       
    def val_one_epoch(self, eval_flag=False):
        t0 = time.time()
        print('Start Validation')
        epoch_step = len(self.val_data_loader)
        if self.local_rank == 0:
            pbar = tqdm(self.val_data_loader, total=epoch_step, postfix=dict, mininterval=0.3, ascii=True, desc="val")
        else:
            pbar = self.val_data_loader
        val_loss = 0
        self.model.eval()

        for batch_i, (images, targets, _) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # 当使用CPU时，跳过GPU相关指令
            if self.device != torch.device("cpu"):
                torch.cuda.synchronize(self.device)
            
            # forward
            with torch.no_grad():
                if eval_flag:
                    self.model.phase = "inference"
                    ps, outs = self.model(images)
                    #---------------------------------------------------------#
                    #   将预测框进行堆叠，然后进行非极大抑制
                    #---------------------------------------------------------#
                    results = non_max_suppression(torch.cat(outs, 1), self.num_classes, 
                                                  conf_thres=self.confidence, nms_thres=self.nms_iou)
                    self.eval_callback.update_batch_statistics(results, targets)
                else:
                    #----------------------#
                    #   前向传播
                    #----------------------#
                    ps = self.model(images)

                #----------------------#
                #   计算损失
                #----------------------#
                loss_value = self.loss(ps, targets)

            val_loss += loss_value.item()

            time_span = time.time() - t0

            if self.local_rank == 0:
                pbar.set_postfix(**{'time(s)':time_span, 
                                    'val_loss': val_loss / (batch_i + 1)})
        # print('Finish Validation')
        return val_loss / epoch_step

    def save_model(self, path):
        print("saving model ...")
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, path)


if __name__ == "__main__":
    print("[%s] Start train ..." % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    Trainer(gpu=0, fp16=True, pretrained=True, distributed=False, Freeze_Train=True).train()

    print("[%s] End train ..." % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    
