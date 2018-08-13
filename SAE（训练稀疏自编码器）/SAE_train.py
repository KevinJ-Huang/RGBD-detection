import tensorflow as tf
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
from skimage import data, exposure, img_as_float
import os

input_nodes = 8 * 8
hidden_size = 100
output_nodes = 8 * 8

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# def sampleImage(file):
#     pic = io.imread(file)
#     shape = pic.shape
#     patchsize = 8
#     numpatches = 1000
#     patches = []
#     i = np.random.randint(0, shape[0] - patchsize, numpatches)
#     j = np.random.randint(0, shape[1] - patchsize, numpatches)
#     k = np.random.randint(0, shape[2], numpatches)
#
#     for l in range(numpatches):
#         temp = pic[i[l]:(i[l] + patchsize), j[l]:(j[l] + patchsize), k[l]]
#         temp = temp.reshape(patchsize * patchsize)
#         patches.append(temp)
#     return patches





def sampleImage(file):
    pic = io.imread(file)

    # pic=Image.fromarray(pic)
    # pic=  pic.convert("I")
    # pic = img_as_float(pic)
    # pic.resize(480,640,1)
    #
    shape = pic.shape
    patchsize = 8
    numpatches = 1000
    patches = []
    i = np.random.randint(0, shape[0] - patchsize, numpatches)
    j = np.random.randint(0, shape[1] - patchsize, numpatches)
    k = np.random.randint(0, shape[2], numpatches)

    for l in range(numpatches):
        temp = pic[i[l]:(i[l] + patchsize), j[l]:(j[l] + patchsize), k[l]]
        temp = temp.reshape(patchsize * patchsize)
        patches.append(temp)
    return patches


def show_image(w):
    sum = np.sqrt(np.sum(w ** 2, 0))
    changedw = w / sum
    a, b = changedw.shape
    c = np.sqrt(a * b)
    d = int(np.sqrt(a))
    e = int(c / d)
    buf = 1
    newimage = -np.ones((buf + (d + buf) * e, buf + (d + buf) * e))
    k = 0
    for i in range(e):
        for j in range(e):
            maxvalue = np.amax(changedw[:, k])
            if (maxvalue < 0):
                maxvalue = -maxvalue
            newimage[(buf + i * (d + buf)):(buf + i * (d + buf) + d),
            (buf + j * (d + buf)):(buf + j * (d + buf) + d)] = np.reshape(changedw[:, k], (d, d)) / maxvalue
            k += 1

    plt.figure("beauty")
    plt.imshow(newimage)
    print(newimage.shape)
    plt.axis('off')
    plt.show()


def computecost(w, b, x, w1, b1):
    p = 0.1
    beta = 3
    lamda = 0.00001

    hidden_output = tf.sigmoid(tf.matmul(x, w) + b)
    pj = tf.reduce_mean(hidden_output, 0)
    sparse_cost = tf.reduce_sum(p * tf.log(p / pj) + (1 - p) * tf.log((1 - p) / (1 - pj)))
    output = tf.sigmoid(tf.matmul(hidden_output, w1) + b1)
    regular = lamda * (tf.reduce_sum(w * w) + tf.reduce_sum(w1 * w1)) / 2
    cross_entropy = tf.reduce_mean(
        tf.pow(output - x, 2)) / 2 + sparse_cost * beta + regular  # + regular+sparse_cost*beta
    return cross_entropy, hidden_output, output


def xvaier_init(input_size, output_size):
    low = -np.sqrt(6.0 / (input_nodes + output_nodes))
    high = -low
    return tf.random_uniform((input_size, output_size), low, high, dtype=tf.float32)


def main():
    w = tf.Variable(xvaier_init(input_nodes, hidden_size))
    b = tf.Variable(tf.truncated_normal([hidden_size], 0.1))
    x = tf.placeholder(tf.float32, shape=[None, input_nodes])
    w1 = tf.Variable(tf.truncated_normal([hidden_size, input_nodes], -0.1, 0.1))
    b1 = tf.Variable(tf.truncated_normal([output_nodes], 0.1))

    cost, hidden_output, output = computecost(w, b, x, w1, b1)
    train_step = tf.train.AdamOptimizer().minimize(cost)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # classes = {'apple', 'ball','banana','bowl','garlic','green','lemon','mushroom','onion','orange','peach','pear','potato','tomato'}
    index=0
    for i in range(100):
        for file in os.listdir('/media/jhm/F86A952A6A94E722/HuangJie/Creat_FRCNN_DataSet-master/rgb1/'):
            index=index+1
            if index%5==0:
                train_x = sampleImage('/media/jhm/F86A952A6A94E722/HuangJie/Creat_FRCNN_DataSet-master/rgb1/'+file)
                _, hidden_output_, output_, cost_, w_ ,b_ = sess.run([train_step, hidden_output, output, cost, w, b],
                                                                 feed_dict={x: train_x})
        if i % 1 == 0:
           print(hidden_output_)
           print(output_)
           print(cost_)
    np.save("weights1_dect.npy", w_)



if __name__ == '__main__':
    # main()
    c = np.load("weights1_dect.npy")

    # show_image(c)
    # print(c[0:8,0])
    print(c.shape)
    # # print(c[:,0])
    c.resize(8,8,1,100)
    d=np.empty([8,8,3,100])
    for i in range(8):
        for j in range(8):
            for l in range(3):
                for k in range(100):
                    d[i,j,l,k]=c[i,j,0,k]
    print(d.shape)
    np.save("w_dect.npy",d)