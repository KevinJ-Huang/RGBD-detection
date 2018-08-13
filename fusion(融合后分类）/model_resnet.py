import os
# ResNet V2
# 载入模块、TensorFlow
import collections
import tensorflow as tf

slim = tf.contrib.slim

# 定义Block
class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    'A named tuple describing a ResNet block'

# 定义降采样subsample方法
def subsample(inputs, factor, scope=None):

    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

# 定义conv2d_same函数创建卷积层
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):

    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                           padding='SAME', scope=scope)
    else:
        # kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           padding='VALID', scope=scope)


@slim.add_arg_scope
# 定义堆叠Blocks函数，两层循环
def stack_blocks_dense(net, blocks,
                       outputs_collections=None):

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    net = block.unit_fn(net,
                                        depth=unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net

# 创建ResNet通用arg_scope，定义函数默认参数值
def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):

    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):

            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


@slim.add_arg_scope
# 定义核心bottleneck残差学习单元
def bottleneck(inputs, depth, depth_bottleneck, stride,
               outputs_collections=None, scope=None):

    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                               scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.name,
                                                output)

# 定义生成ResNet V2的主函数
def resnet_v2(inputs,
              blocks,
              num_classes=None,
              global_pool=True,
              include_root_block=True,
              reuse=None,
              scope=None):

    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck,
                             stack_blocks_dense],
                            outputs_collections=end_points_collection):
            net = inputs
            if include_root_block:

                with slim.arg_scope([slim.conv2d],
                                    activation_fn=None, normalizer_fn=None):
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net = stack_blocks_dense(net, blocks)

            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
            if global_pool:
                # Global average pooling.
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
            if num_classes is not None:
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='logits')
            # Convert end_points_collection into a dictionary of end_points.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if num_classes is not None:
                end_points['predictions'] = slim.softmax(net, scope='predictions')
            logits=tf.squeeze(end_points['predictions'], [1, 2], name='SpatialSqueeze')
            return logits

# 设计层数为50的ResNet V2
def inference50(inputs,batch_size,
                num_classes,
                global_pool=True,
                reuse=None,
                scope='inference50'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)

# 设计101层的ResNet V2
def inference101(inputs,batch_size,
                 num_classes,
                 global_pool=True,
                 reuse=None,
                 scope='inference101'):
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)

# 设计152层的ResNet V2
def inference152(inputs,batch_size,
                 num_classes,
                 global_pool=True,
                 reuse=None,
                 scope='inference152'):
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)

# 设计200层的ResNet V2
def inference200(inputs,batche_size,
                 num_classes,
                 global_pool=True,
                 reuse=None,
                 scope='inference200'):
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)


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
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy
