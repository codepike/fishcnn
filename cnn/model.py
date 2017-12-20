import tensorflow as tf

class CNN(object):
    def __init__(self):
        pass

    def predict(self, images, keep_prob=0.5):
        images = tf.reshape(images, [-1,28,28,1])

        conv1 = conv_layer(images, shape=[5,5,1,32])
        conv1_pool = max_pool_2x2(conv1)

        conv2 = conv_layer(conv1_pool, shape=[5,5,32,64])
        conv2_pool = max_pool_2x2(conv2)

        # conv3 = conv_layer(conv2_pool, shape=[5,5,64,128])
        # conv3_pool = max_pool_2x2(conv3)

        conv3_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
        conv3_drop = tf.nn.dropout(conv3_flat, keep_prob = keep_prob)

        full1 = tf.nn.relu(full_layer(conv3_drop, 512))
        full1_drop = tf.nn.dropout(full1, keep_prob=keep_prob)

        logits = full_layer(full1_drop, 10)

        return logits

    def calculate_loss(self, logits, labels):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        return tf.reduce_mean(loss)

    def build_graph(self, images, labels, keep_prob=0.5):
        logits = self.predict(images, keep_prob)
        return self.calculate_loss(logits, labels)

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
