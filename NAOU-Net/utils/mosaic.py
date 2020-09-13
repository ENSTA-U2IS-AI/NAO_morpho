import numpy as np
from skimage.morphology import erosion, dilation,binary_erosion, opening, closing, white_tophat, reconstruction, area_opening
from skimage.morphology import black_tophat, skeletonize, convex_hull_image,extrema
from skimage.morphology import square, diamond, octagon, rectangle, star, disk, label
from skimage.segmentation import watershed
from PIL import Image
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
from skimage import data, util, filters, color
import math
from skimage import io, color
from matplotlib import pyplot as plt

import os







se1_0 = np.array([[ 0, 1, 0],
              [ 0, 1, 0],
              [ 0, 1, 0]], dtype=np.uint8)


se2_0 = np.array([[ 0, 0, 0],
              [ 1, 1, 1],
              [0, 0, 0]], dtype=np.uint8)


se3_0 = np.array([[ 0, 0, 1],
              [0,  1, 0],
              [1, 0, 0]], dtype=np.uint8)


se4_0 = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]], dtype=np.uint8)


se1_1 = np.array([[0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0]], dtype=np.uint8)


se2_1 = np.array([[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]], dtype=np.uint8)


se3_1 = np.array([[0, 0, 0, 0, 1],
              [0, 0, 0, 1, 0],
              [0, 0, 1, 0, 0],
              [0, 1, 0, 0, 0],
              [1, 0, 0, 0, 0]], dtype=np.uint8)


se4_1 = np.array([[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1]], dtype=np.uint8)


se1_1 = np.array([[0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0]], dtype=np.uint8)


se2_1 = np.array([[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]], dtype=np.uint8)


se3_1 = np.array([[0, 0, 0, 0, 1],
              [0, 0, 0, 1, 0],
              [0, 0, 1, 0, 0],
              [0, 1, 0, 0, 0],
              [1, 0, 0, 0, 0]], dtype=np.uint8)


se4_1 = np.array([[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1]], dtype=np.uint8)



se1_2 = np.array([[0,0,0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0,0, 0],
              [0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0]], dtype=np.uint8)


se2_2 = np.array([[0,0,0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0,0, 0],
              [1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)


se3_2 = np.array([[0,0,0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1,0, 0],
              [0, 0, 0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

se4_2 = np.array([[1,0,0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0,0, 0],
              [0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 1]], dtype=np.uint8)

scale0=[se1_0,se2_0,se3_0,se4_0]
scale1=[se1_1,se2_1,se3_1,se4_1]
scale2=[se1_2,se2_2,se3_2,se4_2]


def gradient(rgb):
    # description
    # input :image rgb
    # output : contour
    rgb = np.asarray(rgb).astype(np.uint8)
    lab = color.rgb2lab(rgb)

    '''print(np.amin(lab[:, :, 0]), np.amax(lab[:, :, 0]))
    print(np.amin(lab[:, :, 1]), np.amax(lab[:, :, 1]))
    print(np.amin(lab[:, :, 2]), np.amax(lab[:, :, 2]))'''
    ##Sobel operator kernels.
    tensor_grad_L=np.zeros((rgb.shape[0],rgb.shape[1],4))
    tensor_grad_a = np.zeros((rgb.shape[0], rgb.shape[1], 4))
    tensor_grad_b = np.zeros((rgb.shape[0], rgb.shape[1], 4))
    for i in range(len(scale0)):
        imgGrad = dilation(lab[:,:,0], scale0[i]) - erosion(lab[:,:,0], scale0[i])
        tensor_grad_L[:, :, i] = imgGrad

        imgGrad = dilation(lab[:,:,1], scale0[i]) - erosion(lab[:,:,1], scale0[i])
        tensor_grad_a[:, :, i] = imgGrad

        imgGrad  = dilation(lab[:,:,2], scale0[i]) - erosion(lab[:,:,2], scale0[i])
        tensor_grad_b[:, :, i] = imgGrad

    grad_L=np.mean(tensor_grad_L,axis=2)
    grad_a = np.mean(tensor_grad_a, axis=2)
    grad_b = np.mean(tensor_grad_b, axis=2)
    grad_scale0=np.maximum(grad_L, grad_a,grad_b)

    tensor_grad_L=np.zeros((rgb.shape[0],rgb.shape[1],4))
    tensor_grad_a = np.zeros((rgb.shape[0], rgb.shape[1], 4))
    tensor_grad_b = np.zeros((rgb.shape[0], rgb.shape[1], 4))
    for i in range(len(scale1)):
        imgGrad = dilation(lab[:,:,0], scale1[i]) - erosion(lab[:,:,0], scale1[i])
        tensor_grad_L[:, :, i] = imgGrad

        imgGrad = dilation(lab[:,:,1], scale1[i]) - erosion(lab[:,:,1], scale1[i])
        tensor_grad_a[:, :, i] = imgGrad

        imgGrad  = dilation(lab[:,:,2], scale1[i]) - erosion(lab[:,:,2], scale1[i])
        tensor_grad_b[:, :, i] = imgGrad

    grad_L=np.mean(tensor_grad_L,axis=2)
    grad_a = np.mean(tensor_grad_a, axis=2)
    grad_b = np.mean(tensor_grad_b, axis=2)
    grad_scale1=np.maximum(grad_L, grad_a,grad_b)

    tensor_grad_L=np.zeros((rgb.shape[0],rgb.shape[1],4))
    tensor_grad_a = np.zeros((rgb.shape[0], rgb.shape[1], 4))
    tensor_grad_b = np.zeros((rgb.shape[0], rgb.shape[1], 4))
    for i in range(len(scale2)):
        imgGrad = dilation(lab[:,:,0], scale2[i]) - erosion(lab[:,:,0], scale2[i])
        tensor_grad_L[:, :, i] = imgGrad

        imgGrad = dilation(lab[:,:,1], scale2[i]) - erosion(lab[:,:,1], scale2[i])
        tensor_grad_a[:, :, i] = imgGrad

        imgGrad  = dilation(lab[:,:,2], scale2[i]) - erosion(lab[:,:,2], scale2[i])
        tensor_grad_b[:, :, i] = imgGrad

    grad_L=np.mean(tensor_grad_L,axis=2)
    grad_a = np.mean(tensor_grad_a, axis=2)
    grad_b = np.mean(tensor_grad_b, axis=2)
    grad_scale2=np.maximum(grad_L, grad_a,grad_b)

    grad=(grad_scale0+grad_scale1+grad_scale2)/3

    return grad

def contrasted_watershed(grad,threshold=0.01):
    grad = grad - np.amin(grad)
    grad = grad / np.amax(grad)
    local_minima = extrema.h_minima(grad, threshold)
    Image.fromarray(255 * local_minima).show('local minima')

    seeds = label(local_minima, connectivity=1)

    local_minima = local_minima.astype(np.uint8)
    labels_waterhed = watershed(grad, seeds)
    return labels_waterhed

def mosaic(rgb,labels_waterhed):
    rgb = np.asarray(rgb).astype(np.float16)
    r,g,b = np.split(rgb, 3, axis=2)
    img_mosaic = np.zeros((rgb.shape[0], rgb.shape[1], 3))
    img_mosaic_r = np.zeros((rgb.shape[0], rgb.shape[1]))
    img_mosaic_g = np.zeros((rgb.shape[0], rgb.shape[1]))
    img_mosaic_b = np.zeros((rgb.shape[0], rgb.shape[1]))
    nb_cluster=np.amax(labels_waterhed)
    for i in range(nb_cluster):
        img_mosaic_r[labels_waterhed == i] = np.mean(r[labels_waterhed == i])
        img_mosaic_g[labels_waterhed == i] = np.mean(g[labels_waterhed==i])
        img_mosaic_b[labels_waterhed == i] = np.mean(b[labels_waterhed==i])


    img_mosaic[:, :, 0] = img_mosaic_r
    img_mosaic[:, :, 1] = img_mosaic_g
    img_mosaic[:, :, 2] = img_mosaic_b
    img_mosaic = np.asarray(img_mosaic).astype(np.uint8)
    return img_mosaic
  

# data_path = str(os.getcwd().split('/utils')[0])+"/data/BSR/BSDS500/data/"
# img_path =  os.path.join(data_path,'images','train','12003.jpg')
# img =  Image.open(img_path)

# grad =gradient(img)
# Image.fromarray(grad).show('img')

# labels_waterhed=contrasted_watershed(grad,threshold=0.05)
# # imagecolor=color.label2rgb(labels_waterhed, np.asarray(img))

# # Image.fromarray((255*imagecolor).astype(np.uint8)).show('LPE')

# img_mosaic=mosaic(img,labels_waterhed)

# Image.fromarray(img_mosaic).show('img_mosaic')
# plt.imshow(img_mosaic)
# plt.title('img_mosaic')
# plt.show()
