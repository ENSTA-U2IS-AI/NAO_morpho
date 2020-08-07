# -*- coding: utf-8 -*-
import numpy as np
from skimage.morphology import skeletonize
import torch
def calculate_f1_score(img_predict,gt):
    img_predict = img_predict.cpu().detach().numpy().astype('int64')
    gt = gt.cpu().detach().numpy().astype('int64')

    TP_num_pixels = np.sum(gt & img_predict)
    FP_num_pixels = np.sum(img_predict & ~gt)
    FN_num_pixels = np.sum(gt & ~img_predict)

    f1_score = 0.
    if TP_num_pixels+FP_num_pixels==0:
      Precision = 0.0
    else:
      Precision = TP_num_pixels / (TP_num_pixels + FP_num_pixels)
    if TP_num_pixels+FN_num_pixels==0:
      Recall = 0.0
    else:
      Recall =  TP_num_pixels / (TP_num_pixels + FN_num_pixels)

    if Precision+Recall==0:
      F1_measure = 0.0
    else:
      F1_measure = 2* Precision * Recall / (Precision + Recall)

    f1_score+=F1_measure
    return f1_score

def evaluation_F1_measure(img_predict,img_GT):
    """
    Args:
      img_predict: input 4D tensor [B,C,H,W] C = 2 B = 2
      target: input 4D tensor [B,C,H,W] C = 2 B = 2
      class: C: number of categories
    Reture: F1 score
    """
    img_predict = torch.nn.functional.softmax(img_predict,1)
    ## with channel=1 we get the img[B,H,W]
    img_predict = img_predict[:,1]
    ## we get an array with floats
    thresholds = np.array(0,1,20)
    f1_scores = []
    for th in thresholds:
      edge = np.where(img_predict<=th,0,1)
      f1_scores.append(calculate_f1_score(edge,img_GT))
    
    return sum(f1_score)/len(thresholds)
