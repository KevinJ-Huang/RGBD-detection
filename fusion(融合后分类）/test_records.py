import os
import numpy as np
import tensorflow as tf
# import input_data
import model1
import create_records as cr
BATCH_SIZE = 30
N_CLASSES = 14
MAX_STEP = 31
learning_rate = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def evaluate():
    '''''Test one image against the saved models and parameters
    '''

    # you need to change the directories to yours.
    #    img_dir = '/home/hjxu/PycharmProjects/01_cats_vs_dogs/222.jpg'
    #    image_array = get_one_img(img_dir)

    with tf.Graph().as_default():

        # you need to change the directories to yours.
        logs_train_dir = 'recordstrain/'
        tfrecords_file = 'tfrecords2'
        train_batch,train_batch_d,train_label_batch = cr.read_and_decode(tfrecords_file,batch_size=BATCH_SIZE)
        train_batch = tf.cast(train_batch, dtype=tf.float32)
        train_batch_d = tf.cast(train_batch_d, dtype=tf.float32)
        train_label_batch = tf.cast(train_label_batch,dtype=tf.int64)
        train_logits = model1.inference(model1,train_batch,BATCH_SIZE,N_CLASSES)
        train_loss = model1.losses(train_logits, train_label_batch)
        train_op = model1.trainning(train_loss, learning_rate)
        train__acc = model1.evaluation(train_logits, train_label_batch)
        m=np.empty([31])
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                for step in np.arange(MAX_STEP):
                     _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
                     if coord.should_stop():
                         break
                     if step%1==0:
                         print('test: train loss = %.2f, train accuracy = %.2f%%'% (tra_loss, tra_acc * 100.0))
                         m[step]=tra_acc
                _, tra_loss, tra_acc= sess.run([train_op, train_loss, train__acc])
            finally:
                coord.request_stop()
            coord.join(threads)
            sess.close( )
            print(np.mean(m))

evaluate()