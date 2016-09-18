import PIL
import pandas
import scipy
import cv2
from scipy import misc
import numpy as np
from PIL import Image
f=open('/home/zshen5/Data/ImageNet2016/release_test/list.txt')
fnames=f.readlines()
f.close()
for fname in fnames:
    fname,ty=fname.split('.')
    origin_image=cv2.imread('/home/zshen5/Data/ImageNet2016/release_test/testing/'+fname+'.jpg',0)#don't change
    img_pred = cv2.imread('/home/zshen5/Data/ImageNet2016/release_test/predictions_camvid100k/img/'+fname+'.png',0)
    outimg = cv2.resize(img_pred, (origin_image.shape[1],origin_image.shape[0]),interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('/home/zshen5/Data/ImageNet2016/release_test/predictions_camvid100k/img_ori_size/'+fname+'.png',outimg)
print 'resize done success'
