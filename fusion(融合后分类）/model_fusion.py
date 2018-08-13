# %%
import tensorflow as tf
import numpy as np

# %%
def inference(images,images_d,batch_size, n_classes):
    '''''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    # conv1, shape = [kernel size, kernel size, channels, kernel numbers]

    with tf.variable_scope('conv_SAE') as scope:
        weights = np.load('w.npy')
        conv = tf.nn.conv2d(images, weights, strides=[1, 2, 2, 1], padding='SAME')
        conv_SAE = tf.nn.sigmoid(conv, name=scope.name)

    with tf.variable_scope('pooling_lrn_SAE') as scope:
        pool_SAE = tf.nn.max_pool(conv_SAE, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                  padding='SAME', name='pooling1_SAE')
        norm_SAE = tf.nn.lrn(pool_SAE, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                             beta=0.75, name='norm_SAE')

    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 100, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm_SAE, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)


    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 256, 512],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[512],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)


    with tf.variable_scope('conv3') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 512, 1024],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[1024],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)

        # pool1 and norm1
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm1')






    with tf.variable_scope('conv_SAE_d') as scope:
        weights = np.load('w_d.npy')
        conv = tf.nn.conv2d(images_d, weights, strides=[1, 2, 2, 1], padding='SAME')
        conv_SAE_d = tf.nn.sigmoid(conv, name=scope.name)

    with tf.variable_scope('pooling1_lrn_SAE_d') as scope:
        pool_SAE_d = tf.nn.max_pool(conv_SAE_d, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                  padding='SAME', name='pooling1_SAE_d')
        norm_SAE_d = tf.nn.lrn(pool_SAE_d, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                             beta=0.75, name='norm_SAE_d')

    with tf.variable_scope('conv1_d') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 100, 1024],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[1024],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm_SAE_d, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1_d = tf.nn.relu(pre_activation, name=scope.name)

        # pool1 and norm1
    with tf.variable_scope('pooling1_lrn_d') as scope:
        pool1_d = tf.nn.max_pool(conv1_d, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling1_d')
        norm1_d = tf.nn.lrn(pool1_d, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm1_d')







    with tf.variable_scope('fusion') as scope:
        # kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
        #                                          stddev=1e-1), name='weights')
        # conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
        #
        # kernel1 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
        #                                          stddev=1e-1), name='weights1')
        # conv1 = tf.nn.conv2d(self1.pool3, kernel1, [1, 1, 1, 1], padding='SAME')
        weights = tf.get_variable('weights',shape=[1],dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        weights1 = tf.get_variable('weights1',shape=[1],dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        conv=tf.multiply(weights,norm1)
        conv1=tf.multiply(weights1,norm1_d)
        biases = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32),
                             trainable=True, name='biases')
        fusion=conv+conv1
        out = tf.nn.bias_add(fusion,biases)
        feature=tf.nn.relu(out, name='fusion')









        # softmax
    with tf.variable_scope('softmax_linear') as scope:
        reshape = tf.reshape(feature, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('softmax_linear',
                                  shape=[dim, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(reshape, weights), biases, name='softmax_linear')

    return softmax_linear


# %%
def losses(logits, labels):
    '''''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]

    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


# %%
def trainning(loss, learning_rate):
    '''''Training ops, the Op returned by this function is what must be passed to
        'sess.run()' call to cause the model to train.

    Args:
        loss: loss tensor, from losses()

    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# %%
def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).
    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy
