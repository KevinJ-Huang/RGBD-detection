import tensorflow as tf
from PIL import Image
from PIL import ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import os
from skimage import data, exposure, img_as_float


classes={'apple','ball','banana','bowl','garlic','greens','mushroom','lemon','onion','orange','peach','pear','potato','tomato'}
data_dir='rgbd_dataset2/'
for name in classes:
    path = data_dir+ name + '/'
    for file in os.listdir(path):
        class_path= path+file
        rgbd=Image.open(class_path)
        rgbd= cv2.cvtColor(np.asarray(rgbd),cv2.COLOR_RGBA2BGRA)
        rgb = np.zeros((480,640,3),dtype=np.uint8)
        disp= np.zeros((480,640,1),dtype=np.uint8)
        rgb[:, :, 2]=rgbd[:, :, 0]
        rgb[:, :, 1]=rgbd[:, :, 1]
        rgb[:, :, 0]=rgbd[:, :, 2]
        # disp=rgbd[:, :, 3]
        # disp=Image.fromarray(disp)
        # disp=disp.convert("I")
        # disp = img_as_float(disp)
        # disp = exposure.adjust_log(disp,10000000)
        # disp = exposure.rescale_intensity(disp,in_range='image',out_range=np.uint8)
        rgb=Image.fromarray(rgb)
        # disp=Image.fromarray(disp)
        # disp=disp.convert('RGB')
        rgb.save('rgb2/'+str(file))
        # disp.save('depth_/'+str(file))

