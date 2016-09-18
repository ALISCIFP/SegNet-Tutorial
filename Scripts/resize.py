import PIL
import pandas
import scipy
import cv2
from scipy import misc
import numpy as np
from PIL import Image

for i in range(1,2001):
    origin_image=cv2.imread('/home/zshen5/Data/ImageNet2016/ADEChallengeData2016/images/validation/ADE_val_0000'+ str('%04d' %i) +'.jpg',0)#don't change
    img_pred = cv2.imread('/home/zshen5/Data/ImageNet2016/ADEChallengeData2016/predictions_camvid100k/img/ADE_val_0000'+ str('%04d' %i) +'.png',0)
    img_anno = cv2.imread('/home/zshen5/Data/ImageNet2016/ADEChallengeData2016/annotations/validation/ADE_val_0000'+ str('%04d' %i) +'.png',0) #don't change
    outimg = cv2.resize(img_pred, (origin_image.shape[1],origin_image.shape[0]),interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('/home/zshen5/Data/ImageNet2016/ADEChallengeData2016/predictions_camvid100k/img_ori_size/ADE_val_0000' + str('%04d' %i) +'.png',outimg)
print 'resize done success'
