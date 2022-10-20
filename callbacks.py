import os

import matplotlib
import scipy.signal
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.utils import bbox_iou, xywh2xyxy

matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from terminaltables import AsciiTable
from tqdm import tqdm


class History():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.loss_path=os.path.join(self.log_dir, "loss")
        self.train_loss = []
        self.val_loss = []

        if not os.path.exists(self.loss_path):
            os.makedirs(self.loss_path)
        self.writer = SummaryWriter(self.loss_path)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_metrics(self, epoch, tag_value_pairs):
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, epoch)

        with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
            f.write(str(tag_value_pairs["val_mAP"]))
            f.write("\n")

    def append_loss(self, epoch, train_loss, val_loss):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.loss_path, "epoch_train_loss.txt"), 'a') as f:
            f.write(str(train_loss))
            f.write("\n")
        with open(os.path.join(self.loss_path, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('train_loss', train_loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.train_loss))

        plt.figure()
        plt.plot(iters, self.train_loss, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.train_loss, num, 3), 
                     'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), 
                     '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")


class EvalCallback():
    def __init__(self, log_dir, class_names, input_shape, iou_threshold):
        self.input_shape = input_shape
        self.class_names = class_names
        self.iou_threshold = iou_threshold
        self.log_dir = log_dir
        self.map_out_path=os.path.join(self.log_dir, "temp_map_out")
        self.sample_metrics = []  # List of tuples (TP, confs, pred)
        self.labels = []

    def zero(self):
        self.sample_metrics = []  # List of tuples (TP, confs, pred)
        self.labels = []

    def get_metrics(self):
        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*self.sample_metrics))]
        precision, recall, AP, f1, ap_class = self.ap_per_class(true_positives, pred_scores, pred_labels, self.labels)
        evaluation_metrics = [
            ("val_precision", precision.mean()),
            ("val_recall", recall.mean()),
            ("val_mAP", AP.mean()),
            ("val_f1", f1.mean()),
        ]
        # Print class APs and mAP
        ap_table = [["Index", "Class name", "AP"]]
        for i, c in enumerate(ap_class):
            ap_table += [[c, self.class_names[c], "%.5f" % AP[i]]]
        print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean()}")
        return evaluation_metrics


    def update_batch_statistics(self, outputs, targets):
        """ 
        Compute true positives, predicted scores and predicted labels per sample 
        """
        # Extract labels
        self.labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])

        batch_metrics = []
        for sample_i in range(len(outputs)):

            if outputs[sample_i] is None:
                continue

            output = outputs[sample_i]
            pred_boxes = output[:, :4]
            pred_scores = output[:, 4]
            pred_labels = output[:, -1]

            true_positives = np.zeros(pred_boxes.shape[0])

            annotations = targets[targets[:, 0] == sample_i][:, 1:]
            target_labels = annotations[:, 0] if len(annotations) else []
            if len(annotations):
                detected_boxes = []
                target_boxes = annotations[:, 1:]

                for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                    # If targets are found break
                    if len(detected_boxes) == len(annotations):
                        break

                    # Ignore if label is not one of the target labels
                    if pred_label not in target_labels:
                        continue

                    iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes, x1y1x2y2=True).max(0)
                    if iou >= self.iou_threshold and box_index not in detected_boxes:
                        true_positives[pred_i] = 1
                        detected_boxes += [box_index]
            batch_metrics.append([true_positives, pred_scores, pred_labels])
        self.sample_metrics += batch_metrics

    def ap_per_class(self, tp, conf, pred_cls, target_cls):
        """ Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
        # Arguments
            tp:    True positives (list).
            conf:  Objectness value from 0-1 (list).
            pred_cls: Predicted object classes (list).
            target_cls: True object classes (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """

        # Sort by objectness
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Find unique classes
        unique_classes = np.unique(target_cls)

        # Create Precision-Recall curve and compute AP for each class
        ap, p, r = [], [], []
        for c in tqdm(unique_classes, desc="Computing AP"):
            i = pred_cls == c
            n_gt = (target_cls == c).sum()  # Number of ground truth objects
            n_p = i.sum()  # Number of predicted objects

            if n_p == 0 and n_gt == 0:
                continue
            elif n_p == 0 or n_gt == 0:
                ap.append(0)
                r.append(0)
                p.append(0)
            else:
                # Accumulate FPs and TPs
                fpc = (1 - tp[i]).cumsum()
                tpc = (tp[i]).cumsum()

                # Recall
                recall_curve = tpc / (n_gt + 1e-16)
                r.append(recall_curve[-1])

                # Precision
                precision_curve = tpc / (tpc + fpc)
                p.append(precision_curve[-1])

                # AP from recall-precision curve
                ap.append(self.compute_ap(recall_curve, precision_curve))

        # Compute F1 score (harmonic mean of precision and recall)
        p, r, ap = np.array(p), np.array(r), np.array(ap)
        f1 = 2 * p * r / (p + r + 1e-16)

        return p, r, ap, f1, unique_classes.astype("int32")


    def compute_ap(self, recall, precision):
        """ Compute the average precision, given the recall and precision curves.
        Code originally from https://github.com/rbgirshick/py-faster-rcnn.

        # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap



