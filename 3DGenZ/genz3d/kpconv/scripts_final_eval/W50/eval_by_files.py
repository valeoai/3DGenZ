import yaml
import numpy as np
import sys
import os
from pathlib import Path

from genz3d.kpconv.utils.metrics import fast_confusion,IoU_from_confusions
from tqdm import tqdm


bias = 0.2
bias_elements = [0,3, 4, 7, 19]
confusion_matrix = np.zeros((20,20))
label_values = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], dtype=np.int32)
home_path = Path.home()
label_path = os.path.join(home_path, "data/semantic_kitti/sequences/08/labels")
pred_prob_path = "../../test/bias_0.2_weight_50/Log_2020-09-15_16-40-46/val_probs"

#Learning map 
learning_map={
  0 : 0,     # "unlabeled"
  1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
  10: 1,     # "car"
  11: 2,     # "bicycle"
  13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
  15: 3,     # "motorcycle"
  16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
  18: 4,    # "truck"
  20: 5,     # "other-vehicle"
  30: 6,    # "person"
  31: 7,     # "bicyclist"
  32: 8,     # "motorcyclist"
  40: 9,     # "road"
  44: 10,    # "parking"
  48: 11,   # "sidewalk"
  49: 12,    # "other-ground"
  50: 13,    # "building"
  51: 14,    # "fence"
  52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
  60: 9,     # "lane-marking" to "road" ---------------------------------mapped
  70: 15,    # "vegetation"
  71: 16,    # "trunk"
  72: 17,    # "terrain"
  80: 18,    # "pole"
  81: 19,    # "traffic-sign"
  99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
  252: 1,    # "moving-car" to "car" ------------------------------------mapped
  253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
  254: 6,    # "moving-person" to "person" ------------------------------mapped
  255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
  256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
  257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
  258: 4,    # "moving-truck" to "truck" --------------------------------mapped
  259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

learning_map_array = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
for k, v in learning_map.items():
    learning_map_array[k] = v



for file_name in tqdm(os.listdir(pred_prob_path)): 
    file_name_raw = file_name[4:-4]
    

    #Load Labels
    frame_labels = np.fromfile(os.path.join(label_path,file_name_raw+".label"), dtype=np.int32)
    sem_labels = frame_labels & 0xFFFF  # semantic label in lower half
    sem_labels = learning_map_array[sem_labels]
    

    #Load probabilities 
    x = np.load(os.path.join(pred_prob_path,file_name))
    x_extend = np.hstack((np.zeros(x.shape[0])[:,None],x))
    pred_raw = x_extend.astype(np.float32) / 255 #Add the ignored row (row zero)
    

    # Bias correction
    elements = np.arange(0,20)
    seen_classes_idx_metric = np.delete(elements, bias_elements).tolist()
    stk_probs_bias = np.zeros(pred_raw.shape)
    stk_probs_bias[:,[ seen_classes_idx_metric]] = 1.0 * bias
    pred_bias = pred_raw - stk_probs_bias
    frame_preds =  np.argmax(pred_bias,axis=1).astype(np.int32)

    #Calculation of fast confusion 
    tmp_confusion = fast_confusion(sem_labels, frame_preds,label_values)
    #Update confusion matrix

    confusion_matrix += tmp_confusion
    
    

C_tot = confusion_matrix
s1 = '\n'
for cc in C_tot:
    for c in cc:
        s1 += '{:7.0f} '.format(c)
    s1 += '\n'
if True:
    print(s1)

# Remove ignored labels from confusions
for l_ind, label_value in reversed(list(enumerate(label_values))):
    if label_value in [0]:
        print("Ignored labels {}".format(label_value))
        C_tot = np.delete(C_tot, l_ind, axis=0)
        C_tot = np.delete(C_tot, l_ind, axis=1)

# Objects IoU
val_IoUs = IoU_from_confusions(C_tot)

# Compute IoUs
mIoU = np.mean(val_IoUs)
s2 = '{:5.2f} | '.format(100 * mIoU)
for IoU in val_IoUs:
    s2 += '{:5.2f} '.format(100 * IoU)
print(s2 + '\n')
