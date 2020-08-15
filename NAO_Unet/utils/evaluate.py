# -*- coding: utf-8 -*-
import numpy as np
from skimage.morphology import skeletonize
import torch

def calculate_f1_score(img_predict,gt):

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
    thresholds = np.linspace(0,1,20)
 
    f1_score = 0.
    img_predict = img_predict.cpu().detach().numpy().astype('float32')
    img_GT = img_GT.cpu().detach().numpy().astype('int32')

    for i in range(img_predict.shape[0]):
      f1_scores = []
      for th in thresholds:
        edge = np.where(img_predict[i]>=th,1,0)
        f1_scores.append(calculate_f1_score(edge,img_GT[i]))
      f1_score += sum(f1_scores)/len(thresholds)
    f1_score = f1_score/img_predict.shape[0]
 
    return f1_score

def evaluation_OIS(img_predict,img_GT):
    """
    Args:
      img_predict: input 4D tensor [B,C,H,W] C = 2 B = 2
      target: input 4D tensor [B,C,H,W] C = 2 B = 2
      class: C: number of categories
    Reture: F1 score
    """
    img_predict = torch.nn.functional.softmax(img_predict, 1)

    ## with channel=1 we get the img[B,H,W]
    img_predict = img_predict[:, 1]

    ## we get an array with floats
    thresholds = np.linspace(0, 1, 20)

    f1_score = 0.
    img_predict = img_predict.cpu().detach().numpy().astype('float32')
    img_GT = img_GT.cpu().detach().numpy().astype('int32')

    for i in range(img_predict.shape[0]):
        f1_scores = []
        for th in thresholds:
            edge = np.where(img_predict[i] >= th, 1, 0)
            f1_scores.append(calculate_f1_score(edge, img_GT[i]))
        f1_score += max(f1_scores)
    OIS = f1_score / img_predict.shape[0]

    return OIS


def calculate_TP_FP_FN(img_predict,gt):

    TP_num_pixels = np.sum(gt & img_predict)
    FP_num_pixels = np.sum(img_predict & ~gt)
    FN_num_pixels = np.sum(gt & ~img_predict)

    return TP_num_pixels,FP_num_pixels,FN_num_pixels

def evaluation_ODS(img_predict,img_GT):
    """
    Args:
      img_predict: input 4D tensor [B,C,H,W] C = 2 B = 2
      target: input 4D tensor [B,C,H,W] C = 2 B = 2
      class: C: number of categories
    Reture: F1 score
    """
    img_predict = torch.nn.functional.softmax(img_predict, 1)

    ## with channel=1 we get the img[B,H,W]
    img_predict = img_predict[:, 1]

    ## we get an array with floats
    thresholds = np.linspace(0, 1, 20)

    f1_score = 0.
    img_predict = img_predict.cpu().detach().numpy().astype('float32')
    img_GT = img_GT.cpu().detach().numpy().astype('int32')
    TP=0
    FP=0
    FN=0

    for i in range(img_predict.shape[0]):
        for th in thresholds:
            edge = np.where(img_predict[i] >= th, 1, 0)
            parameters = calculate_TP_FP_FN(edge,img_GT)
            TP+=parameters[0]
            FP+=parameters[1]
            FN+=parameters[2]
            
    if TP+FP==0:
      Precision = 0.0
    else:
      Precision = TP / (TP + FP)
    if TP+FN==0:
      Recall = 0.0
    else:
      Recall =  TP / (TP + FN)

    if Precision+Recall==0:
      ODS = 0.0
    else:
      ODS = 2* Precision * Recall / (Precision + Recall)

    return ODS

# def evaluation_F1_measure(img_predict,img_GT):
#     """
#     Args:
#       img_predict: input 4D tensor [B,C,H,W] C = 1 B = 2
#       target: input 4D tensor [B,C,H,W] C = 1 B = 2
#       class: C: number of categories
#     Reture: F1 score
#     """
#     img_predict = torch.max(img_predict,1)[1]
#     img_predict = img_predict.cpu().detach().numpy().astype('int64')
#     img_GT = img_GT.cpu().detach().numpy().astype('int64')
#     F1_score = 0.0
#     for i in range(img_predict.shape[0]):
#     # #when batch_size==1
#     # img_predict = torch.squeeze(img_predict.cpu()).detach().numpy()
#     # img_GT = torch.squeeze(img_GT.cpu()).detach().numpy()
#     #  # apply image binarization operation
#     # img_predict = np.where(img_predict<(img_predict.max()-img_predict.min())/2,0,1)
#     # img_GT = np.where(img_GT<(img_GT.max()-img_GT.min())/2,0,1)

#       TP_num_pixels = np.sum(img_GT[i] & img_predict[i])
#       FP_num_pixels = np.sum(img_predict[i] & ~img_GT[i])
#       FN_num_pixels = np.sum(img_GT[i] & ~img_predict[i])

#       if TP_num_pixels+FP_num_pixels==0:
#         Precision = 0.0
#       else:
#         Precision = TP_num_pixels / (TP_num_pixels + FP_num_pixels)
#       if TP_num_pixels+FN_num_pixels==0:
#         Recall = 0.0
#       else:
#         Recall =  TP_num_pixels / (TP_num_pixels + FN_num_pixels)

#       if Precision+Recall==0:
#         F1_measure = 0.0
#       else:
#         F1_measure = 2* Precision * Recall / (Precision + Recall)
#       F1_score+=F1_measure
    
#     return F1_score/img_predict.shape[0]

