import numpy as np
import os
import cv2
from numpy import random
def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

def norm_image(image):
    """
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    cam = 0.5 * heatmap + 0.5 * image
    return norm_image(cam), heatmap

Model="SINet_V2_convnext_L2_edgetask_fusion"
Dataset="COD10K"
oripath=f"/media/perry/E/Model_e/SINet-V2-main/Dataset/TestDataset/{Dataset}/Imgs/"
gtpath=f"/media/perry/E/Model_e/SINet-V2-main/Dataset/TestDataset/{Dataset}/Edge/"
predpath=f"/media/perry/E/Model_e/SINet-V2-main/res/{Model}/{Dataset}/"
savepath=f"/media/perry/E/Model_e/SINet-V2-main/Statistics/compareimg/{Model}/Compare/{Dataset}/"
mkdir(savepath)
orilist=os.listdir(oripath)
gtlist=os.listdir(gtpath)
preflist=os.listdir(predpath)
orilist.sort()
gtlist.sort()
preflist.sort()
for i,(oriimgpath,gtimgpath,predimgpath) in enumerate(zip(orilist,gtlist,preflist)):
    oriimg=cv2.imread(oripath+oriimgpath)
    gtimg=cv2.imread(gtpath+gtimgpath)
    predimg=cv2.imread(predpath+predimgpath)

    gtimg = cv2.cvtColor(gtimg, cv2.COLOR_BGR2GRAY)
    gtimg = cv2.normalize(gtimg.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    predimg = cv2.cvtColor(predimg, cv2.COLOR_BGR2GRAY)
    predimg = cv2.normalize(predimg.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    cam_gt, heatmap_gt=gen_cam(oriimg, gtimg)
    cam_pred, heatmap_pred=gen_cam(oriimg, predimg)

    # cv2.imshow('image0',oriimg)
    # cv2.imshow('image1',cam_gt)
    # cv2.imshow('image2',cam_pred)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(savepath+oriimgpath[:-4]+"_0ori.png", oriimg)
    cv2.imwrite(savepath+gtimgpath[:-4]+"_1gt.png", cam_gt)
    cv2.imwrite(savepath+predimgpath[:-4]+"_2gpred.png", cam_pred)


