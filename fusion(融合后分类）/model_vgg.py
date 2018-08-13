import tensorflow as tf
import numpy as np



def inference(self, images, batch_size, n_classes):
    self.trainable = True
    self.dropout = 0.5
    self.parameters = []
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''

    # conv1_1
    with tf.name_scope('conv1_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        self.conv1_1 = tf.nn.relu(out, name=scope)
        self.parameters += [kernel, biases]

        # conv1_2
    with tf.name_scope('conv1_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        self.conv1_2 = tf.nn.relu(out, name=scope)
        self.parameters += [kernel, biases]

        # pool1
    self.pool1 = tf.nn.max_pool(self.conv1_2,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool1')

    # conv2_1
    with tf.name_scope('conv2_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        self.conv2_1 = tf.nn.relu(out, name=scope)
        self.parameters += [kernel, biases]

        # conv2_2
    with tf.name_scope('conv2_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        self.conv2_2 = tf.nn.relu(out, name=scope)
        self.parameters += [kernel, biases]

        # pool2
    self.pool2 = tf.nn.max_pool(self.conv2_2,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool2')

    # conv3_1
    with tf.name_scope('conv3_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        self.conv3_1 = tf.nn.relu(out, name=scope)
        self.parameters += [kernel, biases]

        # conv3_2
    with tf.name_scope('conv3_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        self.conv3_2 = tf.nn.relu(out, name=scope)
        self.parameters += [kernel, biases]

        # conv3_3
    with tf.name_scope('conv3_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        self.conv3_3 = tf.nn.relu(out, name=scope)
        self.parameters += [kernel, biases]

        # pool3
    self.pool3 = tf.nn.max_pool(self.conv3_3,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool3')

    # conv4_1
    with tf.name_scope('conv4_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        self.conv4_1 = tf.nn.relu(out, name=scope)
        self.parameters += [kernel, biases]

        # conv4_2
    with tf.name_scope('conv4_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        self.conv4_2 = tf.nn.relu(out, name=scope)
        self.parameters += [kernel, biases]

        # conv4_3
    with tf.name_scope('conv4_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        self.conv4_3 = tf.nn.relu(out, name=scope)
        self.parameters += [kernel, biases]

        # pool4
    self.pool4 = tf.nn.max_pool(self.conv4_3,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool4')

    # conv5_1
    with tf.name_scope('conv5_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        self.conv5_1 = tf.nn.relu(out, name=scope)
        self.parameters += [kernel, biases]

        # conv5_2
    with tf.name_scope('conv5_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        self.conv5_2 = tf.nn.relu(out, name=scope)
        self.parameters += [kernel, biases]

        # conv5_3
    with tf.name_scope('conv5_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        self.conv5_3 = tf.nn.relu(out, name=scope)
        self.parameters += [kernel, biases]

        # pool5
    self.pool5 = tf.nn.max_pool(self.conv5_3,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME',
                                name='pool4')

    # fc1
    with tf.name_scope('fc1') as scope:
        shape = int(np.prod(self.pool5.get_shape()[1:]))
        fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                               dtype=tf.float32,
                                               stddev=1e-1), name='weights')
        fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                           trainable=True, name='biases')
        pool5_flat = tf.reshape(self.pool5, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
        self.fc1 = tf.nn.relu(fc1l)
        self.parameters += [fc1w, fc1b]

        # fc2
    with tf.name_scope('fc2') as scope:
        fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                               dtype=tf.float32,
                                               stddev=1e-1), name='weights')
        fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                           trainable=True, name='biases')
        fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
        self.fc2 = tf.nn.relu(fc2l)
        self.parameters += [fc2w, fc2b]

        # fc3
    with tf.name_scope('fc3') as scope:
        fc3w = tf.Variable(tf.truncated_normal([4096, n_classes],
                                               dtype=tf.float32,
                                               stddev=1e-1), name='weights')
        fc3b = tf.Variable(tf.constant(1.0, shape=[n_classes], dtype=tf.float32),
                           trainable=True, name='biases')
        self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
        self.parameters += [fc3w, fc3b]

    return self.fc3l


def losses(logits, labels):
    '''Compute loss from logits and labels
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
    '''Training ops, the Op returned by this function is what must be passed to
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
        print(labels)
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy
