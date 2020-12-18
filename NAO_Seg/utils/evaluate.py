# -*- coding: utf-8 -*-
import numpy as np
from skimage.morphology import skeletonize
import torch
from PIL import Image
import imageio
from matplotlib import pyplot as plt

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
    img_GT = img_GT.cpu().detach().numpy().astype('bool')

    for i in range(img_predict.shape[0]):
      f1_scores = []
      for th in thresholds:
        edge = np.where(img_predict[i]>=th,True,False)
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
    thresholds = np.linspace(0, 1, 100)

    OIS_th = 0.
    img_predict = img_predict.cpu().detach().numpy().astype('float32')
    img_GT = img_GT.cpu().detach().numpy().astype(np.bool)

    for i in range(img_predict.shape[0]):
        f1_scores = []
        for th in thresholds:
            edge = np.where(img_predict[i] >= th, 1, 0).astype(np.bool)
            f1_scores.append(calculate_f1_score(edge, img_GT[i]))
        OIS_th += np.max(np.array(f1_scores))
    OIS = OIS_th / img_predict.shape[0]

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
    thresholds = np.linspace(0, 1, 100)

    f1_score = 0.
    img_predict = img_predict.cpu().detach().numpy().astype('float32')
    img_GT = img_GT.cpu().detach().numpy().astype(np.bool)
    TP=0
    FP=0
    FN=0

    for i in range(img_predict.shape[0]):
        for th in thresholds:
            edge = np.where(img_predict[i] >= th, 1, 0).astype(np.bool)
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

def save_predict_imgs(img_predict,nums):
    """
      Args:
        img_predict: input 4D tensor [B,C,H,W] C = 2 B = ?
        class: C: number of categories
      Reture: F1 score
      """

    # test the img_predict
    img_predict = torch.nn.functional.softmax(img_predict, 1)
    edge = np.zeros((416, 416))
    img_predict = img_predict[:, 1]
    img_predict = img_predict.squeeze()
    img_predict = img_predict.cpu().detach().numpy().astype('float32')
    threshold = 0.76
    img_predict = np.where(img_predict>=threshold,img_predict,0)
    img_predict *= 255
    edge = 255 - img_predict
    edge = Image.fromarray(np.uint8(edge))
    edge = edge.convert('L')
    edge.save('./test/' + str(nums) + '.png')

    return


def accuracyandlossCurve(loss, valid_OIS, epoches):
    """this fonction is to display the accuracy and loss of the validation"""
    x_loss = np.linspace(1, epoches, num=epoches)
    x_accuracy = np.linspace(1, epoches, num=epoches)
    y_loss = loss
    y_valid_OIS = valid_OIS
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(x_loss, y_loss)
    plt.title('Valid: loss Vs epoches')
    plt.ylabel('Valid loss')
    # y_ticks = np.arange(0, 2, 0.2)
    # plt.yticks(y_ticks)

    plt.subplot(2, 1, 2)
    plt.plot(x_accuracy, y_valid_OIS, color='red', label='valid_ois')
    # plt.plot(x_accuracy,y_accuracy_top5,color='blue',label='top5')
    plt.title('Valid: accuracy Vs epoches')
    plt.ylabel('Valid OIS')
    # y_ticks = np.arange(0, 1.1, 0.1)
    # plt.yticks(y_ticks)
    plt.legend()
    plt.savefig('./curve/accuracy_loss_validation.png')
    plt.show()

# def accuracyandlossCurve(loss, valid_OIS, epoches):
#     """this fonction is to display the accuracy and loss of the validation"""
#     x_loss = np.linspace(1, epoches, num=epoches)
#     x_accuracy = np.linspace(1, epoches, num=epoches)
#     y_loss = loss
#     y_valid_OIS = valid_OIS
#     plt.figure(figsize=(10, 8))

#     plt.subplot(2, 1, 1)
#     plt.plot(x_loss, y_loss)
#     plt.title('Valid: loss Vs epoches')
#     plt.ylabel('Valid loss')
#     # y_ticks = np.arange(0, 2, 0.2)
#     # plt.yticks(y_ticks)

#     plt.subplot(2, 1, 2)
#     plt.plot(x_accuracy, y_valid_OIS, color='red', label='train_loss')
#     # plt.plot(x_accuracy,y_accuracy_top5,color='blue',label='top5')
#     plt.title('train: loss Vs epoches')
#     plt.ylabel('train_loss')
#     # y_ticks = np.arange(0, 1.1, 0.1)
#     # plt.yticks(y_ticks)
#     plt.legend()
#     plt.savefig('./curve/loss.png')
#     plt.show()
