import os
import shutil

import matplotlib
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from utils.utils_bbox import non_max_suppression, yolo_correct_boxes

matplotlib.use('Agg')
from config import Config
from matplotlib import pyplot as plt
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.utils_map import get_coco_map, get_map

cfg = Config()

class Evaluater():
    def __init__(self, log_dir, annotation_path, class_names, input_shape, 
                 conf_thres, nms_thres, letterbox_image=False, max_boxes=100, MINOVERLAP=0.5):
        self.input_shape = input_shape
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.class_names = class_names
        self.letterbox_image = letterbox_image
        self.max_boxes = max_boxes
        self.log_dir = log_dir
        self.map_out_path = os.path.join(self.log_dir, "temp_map_out")
        self.annotation_path = ".\\data\\voc_val_data.txt"

        self.MINOVERLAP = MINOVERLAP

        self.maps       = [0]
        self.epoches    = [0]
        self.cuda = torch.cuda.is_available()

        if self.cuda:
            self.device = torch.device("cuda", 0)
        else:
            self.device = torch.device("cpu")

        if not os.path.exists(self.map_out_path):
            os.makedirs(self.map_out_path)
        if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
            os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
        if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
            os.makedirs(os.path.join(self.map_out_path, "detection-results"))

    def get_map_txt(self, net):
        with open(self.annotation_path) as f:
            annotation_lines = f.readlines()

        print("Get map.")
        for annotation_line in tqdm(annotation_lines):
            line        = annotation_line.split()
            image_id    = os.path.basename(line[0]).split('.')[0]
            #------------------------------#
            #   读取图像并转换成RGB图像
            #------------------------------#
            image       = Image.open(line[0])
            #------------------------------#
            #   获得真实框
            #------------------------------#
            gt_boxes    = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])

            #------------------------------#
            #   获得真实框txt
            #------------------------------#
            with open(os.path.join(self.map_out_path, "ground-truth\\" + image_id + ".txt"), "w", encoding='utf-8') as f:
                for box in gt_boxes:
                    left, top, right, bottom, obj = box
                    obj_name = self.class_names[int(obj)]
                    f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

            #------------------------------#
            #   获得预测txt
            #------------------------------#
            with open(os.path.join(self.map_out_path, "detection-results\\" + image_id + ".txt"), "w", encoding='utf-8') as f:
                image_shape = np.array(np.shape(image)[0:2])
                #---------------------------------------------------------#
                #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
                #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
                #---------------------------------------------------------#
                image       = cvtColor(image)
                #---------------------------------------------------------#
                #   给图像增加灰条，实现不失真的resize
                #   也可以直接resize进行识别
                #---------------------------------------------------------#
                crop_img  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
                #---------------------------------------------------------#
                #   添加上batch_size维度
                #---------------------------------------------------------#
                photo = preprocess_input(np.array(crop_img, dtype=np.float32))
                photo = np.transpose(photo, (2, 0, 1))
                #---------------------------------------------------------#
                #   添加上batch_size维度
                #---------------------------------------------------------#
                image_data  = np.expand_dims(photo, 0)

                with torch.no_grad():
                    images = torch.from_numpy(image_data)
                    if self.cuda:
                        net.to(self.device)
                        net.eval()
                        images = images.to(self.device)
                    #---------------------------------------------------------#
                    #   将图像输入网络当中进行预测！
                    #---------------------------------------------------------#
                    outputs, _ = net(images)
                    results = non_max_suppression(torch.cat(outputs, 1), len(self.class_names), 
                                                  conf_thres=self.conf_thres, nms_thres=self.nms_thres)

                    #---------------------------------------------------------#
                    #   如果没有检测出物体，返回原图
                    #---------------------------------------------------------#
                    try :
                        result = results[0].cpu().numpy()
                    except:
                        return False

                    top_label   = np.array(result[:, 6], dtype='int32')
                    top_conf    = result[:, 4] * result[:, 5]
                    top_boxes   = result[:, :4]

                top_100     = np.argsort(top_conf)[::-1][:self.max_boxes]
                top_boxes   = top_boxes[top_100]
                top_conf    = top_conf[top_100]
                top_label   = top_label[top_100]

                top_xy, top_wh = (top_boxes[:, 0:2] + top_boxes[:, 2:4])/2, top_boxes[:, 2:4] - top_boxes[:, 0:2]
                #-----------------------------------------------------------------#
                #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条
                #   因此生成的top_bboxes是相对于有灰条的图像的
                #   我们需要对其进行修改，去除灰条的部分。
                #-----------------------------------------------------------------#
                boxes = yolo_correct_boxes(top_xy, top_wh, self.input_shape,image_shape, self.letterbox_image)

                for i, c in enumerate(top_label):
                    predicted_class = self.class_names[int(c)]
                    score = str(top_conf[i])

                    top, left, bottom, right = boxes[i]
                    f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        return True

    def calculate_map(self, net):
        if self.get_map_txt(net):
            print("Calculate Map.")
            try:
                map_res = get_coco_map(class_names = self.class_names, path = self.map_out_path)[1]
            except:
                map_res = get_map(self.MINOVERLAP, False, path = self.map_out_path)
            return map_res

    def calculate_map_epoch(self, net, epoch):
        temp_map = self.calculate_map(net)
        self.maps.append(temp_map)
        self.epoches.append(epoch)

        with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
            f.write(str(temp_map))
            f.write("\n")
        
        plt.figure()
        plt.plot(self.epoches, self.maps, 'red', linewidth = 2, label='train map')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Map %s'%str(self.MINOVERLAP))
        plt.title('A Map Curve')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
        plt.cla()
        plt.close("all")

        print("Get map done.")
        shutil.rmtree(self.map_out_path)

