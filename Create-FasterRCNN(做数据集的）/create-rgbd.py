import tensorflow as tf
from PIL import Image
from PIL import ImageEnhance
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import cv2
import numpy as np
import os
from skimage import data, exposure, img_as_float


classes={'apple','ball','banana','bowl','garlic','greens','mushroom','lemon','onion','orange','peach','pear','potato','tomato'}
data_dir = 'dataset2/'
data_dir_d='dataset2_d/'
i=0
for name in classes:
    class_path = data_dir + name + '/'
    for img_name in os.listdir(class_path):
        i=i+1
        if i%39==0:
            img_path = class_path + img_name
            img_path_d=data_dir_d+name+ '/'+img_name
            if os.path.exists(img_path):
                rgb=Image.open(img_path)
                rgb = np.array(rgb)
            else:
                continue
            if os.path.exists(img_path_d):
                disp = Image.open(img_path_d)
                disp = np.array(disp)
            else:
                continue
            rgbd = np.zeros((480,640,4),dtype=np.uint8)
            rgbd[:, :, 0] = rgb[:, :, 0]
            rgbd[:, :, 1] = rgb[:, :, 1]
            rgbd[:, :, 2] = rgb[:, :, 2]
            rgbd[:, :, 3] = disp
            im=Image.fromarray(rgbd)
            # c=np.array(im)
            # print(c.shape)
            t=i/39
            if (t>0)and(t<10):
                im.save('rgbd_dataset2/'+str(name)+'/'+'10000'+str(int(t))+'.png')
            if (t>10)and(t<100):
                im.save('rgbd_dataset2/'+str(name)+'/'+'1000'+str(int(t))+'.png')
            if (t>100)and(t<1000):
                im.save('rgbd_dataset2/'+str(name)+'/'+'100'+str(int(t))+'.png')
            if (t>1000)and(t<10000):
                im.save('rgbd_dataset2/'+str(name)+'/'+'10'+str(int(t))+'.png')
            if (t>10000)and(t<100000):
                im.save('rgbd_dataset2/'+str(name)+'/'+'1'+str(int(t))+'.png')
            if (t>100000)and(t<1000000):
                im.save('rgbd_dataset2/'+str(name)+'/'+str(int(t))+'.png')


# disp=Image.open('png/apple_d.png')
# # plt.imshow(im)
# # plt.show()
# # rgbd= cv2.cvtColor(np.asarray(im),cv2.COLOR_RGBA2BGRA)
# # rgb = np.zeros((480,640,3),dtype=np.uint8)
# # disp= np.zeros((480,640,1),dtype=np.uint8)
# # rgb[:, :, 0]=rgbd[:, :, 0]
# # rgb[:, :, 1]=rgbd[:, :, 1]
# # rgb[:, :, 2]=rgbd[:, :, 2]
# # disp=rgbd[:, :, 0]
# # disp=Image.fromarray(disp)
# disp=disp.convert("I")
# # plt.imshow(disp)
# # plt.show()
# img_d = img_as_float(disp)
# img_d = exposure.adjust_log(img_d,100000000)
# plt.imshow(img_d)
# plt.show()
# print(img_d)