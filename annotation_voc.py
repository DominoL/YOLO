"""
本脚本有两个功能：
1.将voc数据集标注信息(.xml)转为yolo标注格式(.txt)，并将图像文件复制到相应文件夹
2.根据json标签文件, 生成对应names标签(pascal_voc_label.names)
3.创建data.data文件, 记录classes个数, train以及val数据集文件(.txt)路径和label.names文件路径
"""
import enum
import json
import os
import shutil

from lxml import etree
from tqdm import tqdm

# voc数据集根目录以及版本
voc_root = ".\\data_open\\VOCdevkit"
voc_version = "VOC2012"

# 转换的数据集以及验证集对应txt文件
train_txt= "train.txt"
val_txt = "val.txt"

# 转换后的文件保存目录
save_file_root = ".\\data"

# label标签对应json文件
label_json_path = ".\\data\\pascal_voc_classes.json"

# label名称
classes_path = ".\\data\\voc_classes.names"

# 拼接出voc的images目录，xml目录，txt目录
voc_images_path = os.path.join(voc_root, voc_version, "JPEGImages")
voc_xml_path = os.path.join(voc_root, voc_version, "Annotations")
train_txt_path = os.path.join(voc_root, voc_version, "ImageSets", "Main", train_txt)
val_txt_path = os.path.join(voc_root, voc_version, "ImageSets", "Main", val_txt)

# 检查文件/文件夹都是否存在
assert os.path.exists(voc_images_path), "VOC images path not exist..."
assert os.path.exists(voc_xml_path), "VOC xml path not exist..."
assert os.path.exists(train_txt_path), "VOC train txt file not exist..."
assert os.path.exists(val_txt_path), "VOC val txt file not exist..."
assert os.path.exists(label_json_path), "label_json_path does not exist..."

if os.path.exists(save_file_root) is False:
    os.makedirs(save_file_root)

def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式, 参考tensorflow的recursive_parse_xml_to_dict
    Args:
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """
    if len(xml) == 0:   # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result: # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []

            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

def translate_info(file_names: list, save_root:str, class_dict:dict, train_val='train'):
    """
    将对应xml文件信息转为yolo中使用的txt文件信息
    :param file_names:
    :param save_root:
    :param class_dict:
    :param train_val:
    :return:
    """
    with open(os.path.join(save_root, "voc_{}_data.txt".format(train_val)), "w") as f:
        for file in tqdm(file_names, desc="translate {} file...".format(train_val)):
            # 检查下图像文件是否存在
            img_path = os.path.join(voc_images_path, file + ".jpg")
            assert os.path.exists(img_path), "file:{} not exist...".format(img_path)

            # 检查xml文件是否存在
            xml_path = os.path.join(voc_xml_path, file + ".xml")
            assert os.path.exists(xml_path), "file:{} not exist...".format(xml_path)

            # real xml
            with open(xml_path) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = parse_xml_to_dict(xml)["annotation"]
            img_height = int(data["size"]["height"])
            img_width = int(data["size"]["width"])

            # write object info into txt
            assert "object" in data.keys(), "file: '{} lack of object key.".format(xml_path)
            if len(data["object"]) == 0:
                # 如果xml文件中没有目标就直接忽略该样本
                print("Warning: in '{}' xml, there are no objects.".format(xml_path))
                continue
            
            info = [img_path]
            for obj in data["object"]:
                # 获取每个object的box信息
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                class_name = obj["name"]
                class_index = class_dict[class_name] - 1  # 目标id从0开始

                # 进一步检查数据
                if xmax <= xmin or ymax <= ymin:
                    print("Warning: in '{}' xml, three are some bbox w/h <= 0".format(xml_path))
                    continue

                info.append(",".join([str(i) for i in [xmin, ymin, xmax, ymax, class_index]]))
            f.write(" ".join(info) + "\n")
        
def create_class_names(class_dict: dict):
    keys = class_dict.keys()
    with open(classes_path, "w") as w:
        for index, k in enumerate(keys):
            if index + 1 == len(keys):
                w.write(k)
            else:
                w.write(k + "\n")

def create_data_data(create_data_path, classes_path, train_path, val_path, classes_info):
    # create my_data.data file that record classes, train, valid and names info.
    with open(create_data_path, "w") as w:
        w.write("classes={}".format(len(classes_info)) + "\n")  # 记录类别个数
        w.write("train={}".format(train_path) + "\n")           # 记录训练集对应txt文件路径
        w.write("valid={}".format(val_path) + "\n")             # 记录验证集对应txt文件路径
        w.write("names={}".format(classes_path) + "\n")          # 记录label.names文件路径


def main():
    # read class_dict
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)

    # 读取train.txt中的所有行信息，删除空行
    with open(train_txt_path, "r") as r:
        train_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    # voc信息转yolo，并将图像文件复制到相应文件夹
    translate_info(train_file_names, save_file_root, class_dict, "train")

    # 读取val.txt中的所有行信息，删除空行
    with open(val_txt_path, "r") as r:
        val_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    # voc信息转yolo，并将图像文件复制到相应文件夹
    translate_info(val_file_names, save_file_root, class_dict, "val")

    # 创建data_label.names文件
    create_class_names(class_dict)

    classes_info = [line.strip() for line in open(classes_path, "r").readlines() if len(line.strip()) > 0]
    # 创建data.data文件，记录classes个数, train以及val数据集文件(.txt)路径和label.names文件路径
    create_data_data(".\\data\\voc_data.data", classes_path, 
                     os.path.join(save_file_root, "voc_{}_data.txt".format("train")), 
                     os.path.join(save_file_root, "voc_{}_data.txt".format("val")), 
                     classes_info)


if __name__ == "__main__":
    main()

        