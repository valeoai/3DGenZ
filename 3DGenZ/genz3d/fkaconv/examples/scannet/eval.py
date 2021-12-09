#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--predfolder", "-p", required=True)
parser.add_argument("--area", type=int, default=None)
args = parser.parse_args()
print(args)



gt_label_filenames = []
pred_label_filenames = []
PRED_DIR = os.listdir(args.predfolder)

if args.area is not None:
    PRED_DIR = ["Area_{}".format(args.area)]

for area in PRED_DIR:
    path_gt_label = os.path.join(args.predfolder, area, 'label.npy')
    path_pred_label = os.path.join(args.predfolder, area, 'pred.txt')
    pred_label_filenames.append(path_pred_label)
    gt_label_filenames.append(path_gt_label)

num_room = len(gt_label_filenames)

print(num_room)
print(len(pred_label_filenames))
assert num_room == len(pred_label_filenames)

gt_classes = [0 for _ in range(21)]
positive_classes = [0 for _ in range(21)]
true_positive_classes = [0 for _ in range(21)]
confusion_matrix = np.zeros((21, 21))


for i in range(num_room):
    print(i, "/"+str(num_room))
    print(pred_label_filenames[i])
    pred_label = np.loadtxt(pred_label_filenames[i])
    gt_label = np.load(gt_label_filenames[i])
    for j in range(gt_label.shape[0]):
        gt_l = int(gt_label[j])
        pred_l = int(pred_label[j])
        gt_classes[gt_l] += 1
        positive_classes[pred_l] += 1
        true_positive_classes[gt_l] += int(gt_l == pred_l)
        confusion_matrix[gt_l, pred_l] = confusion_matrix[gt_l, pred_l] +1


#Print and save results
result_path = os.path.join(args.predfolder, "results.txt")
print("Result path {}".format(result_path))
with open(result_path, "w+") as file:
    print(gt_classes)
    print(positive_classes)
    print(true_positive_classes)
    file.write("gt_classes: {}\n".format(gt_classes))
    file.write("positive_classes: {}\n".format(positive_classes))
    file.write("true_positive_classes: {}\n".format(true_positive_classes))

    print('Overall accuracy: {0}'.format(sum(true_positive_classes)/float(sum(positive_classes))))
    file.write("Overall_acc:: {}\n".format(sum(true_positive_classes)/float(sum(positive_classes))))
    file.write("IoU:\n")
    print('IoU:')
    iou_list = []
    for i in range(21):
        iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i])
        print(iou)
        file.write(" {}\n".format(iou))
        iou_list.append(iou)

    print("mIoU: {}".format(sum(iou_list)/21.0))
    file.write("mIoU: {}\n".format(sum(iou_list)/21.0))

    print("Confusion matrix: {}".format(confusion_matrix))
    file.write("Confusoin matrix: {}\n".format(confusion_matrix))
