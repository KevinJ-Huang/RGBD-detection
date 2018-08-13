import os
import tensorflow as tf
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, exposure, img_as_float
#生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

#生成字符串类型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))


# 制作TFRecord格式
def createTFRecord(filename,mapfile,num):
    class_map = {}
    data_dir = 'dataset/'
    data_dir_d='dataset_d/'
    classes = {'apple', 'ball','banana','bowl','garlic','green','lemon','mushroom','onion','orange','peach','pear','potato','tomato'}
    # 输出TFRecord文件的地址
    writer = tf.python_io.TFRecordWriter(filename)
    i=0
    for index, name in enumerate(classes):
        class_path = data_dir + name + '/'
        class_map[index] = name
        for img_name in os.listdir(class_path):
            i+=1
            if (i%num==0)and(i%(num*10)!=0):
                img_path = class_path + img_name  # 每个图片的地址
                img_path_d=data_dir_d+ name + '/'+img_name
                img = Image.open(img_path)
                img = img.resize((208, 208))
                img=img.convert("RGB")
                img_raw = img.tobytes()# 将图片转化
            # 成二进制格式

                img_d= Image.open(img_path_d)
                img_d = img_d.resize((208, 208))
                img_d=img_d.convert("I")
                img_d = img_as_float(img_d)
                img_d = exposure.adjust_gamma(img_d,1.0)
                img_d = exposure.adjust_log(img_d,100000000)
                img_d = exposure.rescale_intensity(img_d,in_range='image',out_range=np.uint8)
                img_raw_d = img_d.tobytes()

                example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(index),
                'image_raw': _bytes_feature(img_raw),
                'image_raw_d':_bytes_feature(img_raw_d)
                }))
                writer.write(example.SerializeToString())
    writer.close()

    txtfile = open(mapfile,'w+')
    for key in class_map.keys():
        txtfile.writelines(str(key)+":"+class_map[key]+"\n")
    txtfile.close()


#读取train.tfrecord中的数据
def read_and_decode(filename):
    #创建一个reader来读取TFRecord文件中的样例
    reader = tf.TFRecordReader()
    #创建一个队列来维护输入文件列表
    filename_queue = tf.train.string_input_producer([filename], shuffle=False,num_epochs = 1)
    #从文件中读出一个样例，也可以使用read_up_to一次读取多个样例
    _,serialized_example = reader.read(filename_queue)
    #     print _,serialized_example
    #解析读入的一个样例，如果需要解析多个，可以用parse_example
    features = tf.parse_single_example(
        serialized_example,
        features = {'label':tf.FixedLenFeature([], tf.int64),
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'image_raw_d': tf.FixedLenFeature([], tf.string),
                    })
    #将字符串解析成图像对应的像素数组
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img,[208, 208, 3]) #reshape为128*128*3通道图片
    img = tf.image.per_image_standardization(img)

    img_d = tf.decode_raw(features['image_raw_d'], tf.uint8)
    img_d = tf.reshape(img_d,[208, 208, 8]) #reshape为128*128*3通道图片
    img_d = tf.image.per_image_standardization(img_d)
    labels = tf.cast(features['label'], tf.int32)
    return img,img_d,labels


def createBatch(filename, batchsize):
    images,images_d,labels = read_and_decode(filename)
    min_after_dequeue = 5
    capacity = min_after_dequeue + 3 * batchsize

    image_batch,image_batch_d,label_batch = tf.train.shuffle_batch([images,images_d,labels],
                                                      batch_size=batchsize,
                                                      capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue
                                                      )

    label_batch = tf.one_hot(label_batch, depth=14)
    return image_batch,image_batch_d,label_batch



if __name__ =="__main__":
    #训练图片两张为一个batch,进行训练，测试图片一起进行测试
    mapfile = "tfrecord.txt"
    train_filename = "tfrecords"
    createTFRecord(train_filename, mapfile,5)
    test_filename="tfrecords2"
    createTFRecord(test_filename,mapfile,50)
    image_batch,image_batch_d,label_batch = createBatch(filename=train_filename, batchsize=3)
    test_images,test_images_d,test_labels = createBatch(filename=test_filename, batchsize=1)

    with tf.Session() as sess:
        initop = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(initop)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while 1:
                _image_batch,_image_batch_d, _label_batch = sess.run([image_batch,image_batch_d,label_batch])
                step += 1
                print(step)
                print(_label_batch)
        except tf.errors.OutOfRangeError:
            print(" trainData done!")

        try:
            step = 0
            while 1:
                _test_images,_test_images_d, _test_labels = sess.run([test_images,test_images_d,test_labels])
                step += 1
                print(step)
                print( _image_batch.shape)
                print( _image_batch_d.shape)
                print(_test_labels)
        except tf.errors.OutOfRangeError:
            print(" TEST done!")
        coord.request_stop()
        coord.join(threads)
