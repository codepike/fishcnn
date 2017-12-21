import tensorflow as tf

class CNN(object):
    """A plain convolution neural network.
    Args:
        reader: a reader providing training data
        is_training: is training
        keep_prob: keep probability
    """
    def __init__(self,  reader, is_training=True, keep_prob=0.5):
        self.reader = reader
        self.keep_prob = keep_prob
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x, self.labels = self.reader.read()
            self.build_graph(keep_prob=self.keep_prob)
            # todo add onehot here

            if is_training:
                self.calc_loss()
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
            else:
                self.calc_accuracy()

    def calc_accuracy(self):
        preditions = tf.argmax(self.logits, 1)
        labels = tf.argmax(self.labels, 1)

        self.acc, self.acc_op = tf.metrics.accuracy(labels=labels, predictions=preditions)

    def build_graph(self, keep_prob=0.5):
        with tf.variable_scope('Conv_layer'):
            # 28 x 28 x 1
            images = tf.reshape(self.x, [-1,28,28,1])

            # 28 x 28 x 32
            conv1 = conv_layer(images, shape=[5,5,1,32])  #

            # 14 x 14 x 32
            conv1_pool = max_pool_2x2(conv1)

            # 14 x 14 x 64
            conv2 = conv_layer(conv1_pool, shape=[5,5,32,64])

            # 7 x 7 x 64
            conv2_pool = max_pool_2x2(conv2)

            # conv3 = conv_layer(conv2_pool, shape=[5,5,64,128])
            # conv3_pool = max_pool_2x2(conv3)

            # full layer 3316
            conv3_flat = tf.reshape(conv2_pool, [-1, 7*7*64])

            # drop 3316
            conv3_drop = tf.nn.dropout(conv3_flat, keep_prob = keep_prob)

            # 512
            full1 = tf.nn.relu(full_layer(conv3_drop, 512))

            # drop 512
            full1_drop = tf.nn.dropout(full1, keep_prob=keep_prob)

            # 10
            self.logits = full_layer(full1_drop, 10)
            self.softmax = tf.nn.softmax(self.logits)

    def calc_loss(self):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        self.loss = tf.reduce_mean(loss)


# util functions here
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                          strides=[1,2,2,1], padding="SAME")


def conv_layer(input, shape):
    w = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, w)+b)


def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    w = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, w) + b
