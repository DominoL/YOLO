import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.data_augumentation import Compose, RandomAugmentHSV, RandomFlip
from utils.utils import preprocess_input


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, transform, multiscale=False, phase='train'):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        # 图像原始形状
        self.shapes = np.array([None] * len(annotation_lines))
        # 原始label: [x1, y1, x2, y2, class] 其中的xyxy都为绝对值
        self.labels = []
        self.input_shape = input_shape
        self.length = len(self.annotation_lines)
        self.transform = transform  #图像的变形处理
        self.phase = phase  #指定train或val

        self.multiscale = multiscale
        self.min_size = input_shape[0] - 3 * 32
        self.max_size = input_shape[0] + 3 * 32
        self.batch_count = 0

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        image, boxes, img_path = self.load_data(index)
        image, boxes = self.letterBox(image, boxes)
        # 数据增强
        image, boxes = self.random_aug_data(image,
                                            boxes,
                                            self.input_shape[0:2],
                                            phase=self.phase)
                                            
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        boxes = np.array(boxes, dtype=np.float32)
        # 将box信息转换到yolo格式
        label = None
        if len(boxes) != 0:
            label = np.zeros((len(boxes), 6))
            # 归一化
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / self.input_shape[1]
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / self.input_shape[0]

            label[:, 1] = boxes[:, 4]  # class
            label[:, 4:6] = boxes[:, 2:4] - boxes[:, 0:2]  # wh
            label[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4] / 2  # center

        return image, label, img_path

    def load_data(self, index):
        line = self.annotation_lines[index].split()
        #------------------------------#
        #   1.读取图像并转换成RGB图像 W*H*C
        #------------------------------#
        img_path = line[0]
        # 将图像转换成RGB图像, 防止灰度图在预测时报错。
        image = Image.open(img_path).convert('RGB')

        #------------------------------#
        #   2.获得预测框
        #------------------------------#
        boxes = np.array(
            [np.array(list(map(float, box.split(',')))) for box in line[1:]])

        return image, boxes, img_path

    def letterBox(self, image, boxes):
        """
        将图像多余的部分加上灰条
        """
        iw, ih = image.size
        h, w = self.input_shape

        # 保证原图比例不变，将图像最大边缩放到指定大小
        ratio = min(w / iw, h / ih)
        nw = int(iw * ratio)
        nh = int(ih * ratio)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        image = image.resize((nw, nh), Image.BICUBIC)
        new_img = Image.new('RGB', (w, h), (128, 128, 128))
        new_img.paste(image, (dx, dy))
        image = new_img

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(boxes) > 0:
            np.random.shuffle(boxes)
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * nw / iw + dx  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * nh / ih + dy  # y1, y2
            boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
            boxes[:, 2][boxes[:, 2] > w] = w
            boxes[:, 3][boxes[:, 3] > h] = h
            boxes_w = boxes[:, 2] - boxes[:, 0]
            boxes_h = boxes[:, 3] - boxes[:, 1]
            boxes = boxes[np.logical_and(boxes_w > 1, boxes_h > 1)]  # 验证无效box

        return image, boxes

    def random_aug_data(self, image, boxes, input_shape, phase):
        #---------------------------------------------------#
        #   3.数据增强
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        image, boxes, input_shape = self.transform(image, boxes, input_shape,
                                                   phase)
        return image, boxes

    def collate_fn(self, batch):
        # DataLoader中collate_fn使用
        images, labels, paths = list(zip(*batch))
        # Remove empty placeholder labels
        labels = [
            torch.from_numpy(label).type(torch.FloatTensor)
            for label in labels if label is not None
        ]
        # Add sample index to labels
        for i, label in enumerate(labels):
            label[:, 0] = i
        labels = torch.cat(labels, 0)

        # Selects new input_shape every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            new_input_shape = np.random.choice(range(self.min_size[0], self.max_size[0] + 1, 32))
            self.input_shape[0] = new_input_shape
            self.input_shape[1] = new_input_shape
        self.batch_count += 1

        images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)

        return images, labels, paths


class DataTransform():
    """
    图像和标注的预处理类。训练和推测时分别采用不同的处理
    学习时进行数据增强处理

    Attributes
    ----------
    input_size : int 需要调整的图像大小。
    color_mean : (B, G, R) 各个颜色通道的平均值。
    """
    def __init__(self, hue=.1, sat=0.7, val=0.4):
        self.data_transform = {
            'train':
            Compose([
                RandomFlip(),  # 翻转图像
                RandomAugmentHSV(hue, sat, val)  # 对图像进行色域变换
            ]),
            'val':
            Compose([])
        }

    def __call__(self, image, box, input_shape, phase):
        """
        :param phase: 'train' or 'val' 指定预处理的模式
        """
        return self.data_transform[phase](image, box, input_shape)
