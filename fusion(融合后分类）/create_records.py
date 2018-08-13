import cv2
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io as io
from skimage import transform


def get_files(file_dir):
    '''''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split('.')
        if name[0]=='cat':
            cats.append(file_dir + file)

            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))

    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]


    return image_list, label_list




def read_and_decode(tfrecords_file,batch_size):
    '''''read and decode tfrecord file, generate (image, label) batches
    Args:
        tfrecords_file: the directory of tfrecord file
        batch_size: number of images in each batch
    Returns:
        image: 4D tensor - [batch_size, width, height, channel]
        label: 1D tensor - [batch_size]
    '''
    # make an input queue from the tfrecord file
    filename_queue = tf.train.string_input_producer([tfrecords_file],shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'image_raw_d': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    image_d = tf.decode_raw(img_features['image_raw_d'], tf.uint8)



    ##########################################################
    # you can put data augmentation here, I didn't use it
    ##########################################################
    # all the images of notMNIST are 28*28, you need to change the image size if you use other dataset.

    image = tf.reshape(image, [208, 208, 3])
    image_d = tf.reshape(image_d, [208, 208, 8])
    label = tf.cast(img_features['label'], tf.float32)
    image = tf.image.per_image_standardization(image)
    image_d=tf.image.per_image_standardization(image_d)
    # image_batch,image_batch_d,label_batch = tf.train.batch([image,image_d,label],
    #                                                         batch_size=batch_size,
    #                                                         num_threads=64,
    #                                                         capacity=2000)

    image_batch,image_batch_d,label_batch = tf.train.shuffle_batch([image,image_d,label],
                                                                   batch_size=batch_size,
                                                                   num_threads=64,
                                                                   capacity=2000,min_after_dequeue=5)
    # label_batch = tf.one_hot(label_batch, depth=14)
    return image_batch,image_batch_d,tf.reshape(label_batch, [batch_size])


# def read_and_decode(filename):
#     #创建一个reader来读取TFRecord文件中的样例
#     reader = tf.TFRecordReader()
#     #创建一个队列来维护输入文件列表
#     filename_queue = tf.train.string_input_producer([filename], shuffle=False,num_epochs = 1)
#     #从文件中读出一个样例，也可以使用read_up_to一次读取多个样例
#     _,serialized_example = reader.read(filename_queue)
#     #     print _,serialized_example
#     #解析读入的一个样例，如果需要解析多个，可以用parse_example
#     features = tf.parse_single_example(
#         serialized_example,
#         features = {'label':tf.FixedLenFeature([], tf.int64),
#                     'image_raw': tf.FixedLenFeature([], tf.string),
#                     'image_raw_d': tf.FixedLenFeature([], tf.string),
#                     })
#     #将字符串解析成图像对应的像素数组
#     img = tf.decode_raw(features['image_raw'], tf.uint8)
#     img = tf.reshape(img,[208, 208, 3]) #reshape为128*128*3通道图片
#     img = tf.image.per_image_standardization(img)
#
#     img_d = tf.decode_raw(features['image_raw_d'], tf.uint8)
#     img_d = tf.reshape(img_d,[208, 208, 3]) #reshape为128*128*3通道图片
#     img_d = tf.image.per_image_standardization(img_d)
#     labels = tf.cast(features['label'], tf.int32)
#     return img,img_d,labels
#
#
# def createBatch(filename, batchsize):
#     images,images_d,labels = read_and_decode(filename)
#     min_after_dequeue = 5
#     capacity = min_after_dequeue + 3 * batchsize
#
#     image_batch,image_batch_d,label_batch = tf.train.shuffle_batch([images,images_d,labels],
#                                                                    batch_size=batchsize,
#                                                                    capacity=capacity,
#                                                                    min_after_dequeue=min_after_dequeue
#                                                                    )
#
#     label_batch = tf.one_hot(label_batch, depth=6)
#     return image_batch,image_batch_d,label_batch




# tfrecords_file = 'tfrecords'
# batch_size=100
# read_and_decode(tfrecords_file)