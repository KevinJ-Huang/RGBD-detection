import tensorflow as tf
from PIL import Image
from PIL import ImageEnhance
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import cv2
import numpy as np
from skimage import data, exposure, img_as_float
# a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
# b = tf.constant([2], shape=[1])
# c=tf.multiply(b,a)
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#
#     print('c=', c.eval())
#     print(c.shape)
#




# image = Image.open('ball.png')
# image=image.convert("L")
# image.show()
# image = img_as_float(image)
# gam2= exposure.adjust_gamma(image, 0.5)
# plt.imshow(gam2,plt.cm.gray)
# enh_col = ImageEnhance.Color(image)
# color = 1.5
# image_colored = enh_col.enhance(color)
# image_colored.show()



# image = Image.open('apple.jpg')
# image=image.convert("I")
# image.show()
# enh_bri = ImageEnhance.Brightness(image)
# brightness = 1.5
# image_brightened = enh_bri.enhance(brightness)
# image_brightened.show()



# lena = mpimg.imread('ball.png')
# # 读取和代码处于同一目录下的 lena.png
# # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
#
#
# plt.imshow(lena) # 显示图片
# plt.axis('off') # 不显示坐标轴
# plt.show()


# img = cv2.imread('ball.png')
# img = Image.open('apple.png')
# img=img.convert("I")
# img = img_as_float(img)
# img = exposure.adjust_gamma(img,1.0)
# img = exposure.adjust_log(img,100000000)
# img = exposure.rescale_intensity(img,in_range='image',out_range=np.uint8)
# result=exposure.is_low_contrast(img)
# print(result)
# print(img.shape)
# plt.imshow(img)
# plt.show()


# img =  Image.open('apple.png')
# img1 =  Image.open('apple1.png')
# img1=   img.convert("RGB")
# # img2=Image.blend(img1, img, 1.0)
# plt.imshow(img)
# plt.show()



import os, os.path

"""将RGB图片和视差图融合成四通道RGBD图片。RGB图片和视差图的路径分别放在txt文件中"""

#通过txt文件中的路径读取RGB图片


rgb = Image.open('apple.png')
disp = Image.open('apple1.png')
rgb = np.array(rgb)
disp = np.array(disp)
# plt.imshow(disp)
# plt.show()
rgbd = np.zeros((480,640,4),dtype=np.uint8)
rgbd[:, :, 0] = rgb[:, :, 0]
rgbd[:, :, 1] = rgb[:, :, 1]
rgbd[:, :, 2] = rgb[:, :, 2]
rgbd[:, :, 3] = disp

im=Image.fromarray(rgbd)
im.save('rgbd.png')
# m=Image.open('rgbd.png')