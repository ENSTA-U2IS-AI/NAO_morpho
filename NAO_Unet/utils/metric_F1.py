# -*- coding: utf-8 -*-
import numpy as np
from skimage.morphology import skeletonize

def evaluation_F1_measure(img_predict,img_GT):
    # apply image binarization operation
    img_predict = np.where(img_predict<(img_predict.max()-img_predict.min())/2,0,1)
    img_GT = np.where(img_GT<(img_GT.max()-img_GT.min())/2,0,1)

    img_GT_skel = skeletonize(img_GT)
    img_predict_skel = skeletonize(img_predict)
    TP_num_pixels = np.sum(img_GT & img_predict)
    FP_num_pixels = np.sum(img_predict_skel & ~img_GT)
    FN_num_pixels = np.sum(img_GT_skel & ~img_predict)

    Precision = TP_num_pixels / (TP_num_pixels + FP_num_pixels)
    Recall =  TP_num_pixels / (TP_num_pixels + FN_num_pixels)

    F1_measure = 2* Precision * Recall / (Precision + Recall)

    return F1_measure
