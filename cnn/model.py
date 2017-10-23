import tensorflow as tf


class CNN(object):
    def __init__(self):
        pass


    def build_graph(self, x, y):
        w = tf.Variable(2.0, tf.float32)
        b = tf.Variable(19.0, tf.float32)

        pred = x * w + b

        diff = pred-y

        loss = tf.reduce_mean(tf.square(diff))

        return loss,w,b
