import numpy as np
import tensorflow as tf

import model
import task
from model import CNN
from task import train, evaluate
from task import TFReader

flags = tf.app.flags

# inputs and outputs
flags.DEFINE_string("input_path", "./train.tfrecord", "The input file path")
flags.DEFINE_string("logdir", "./logdir", "The output path")

# training
flags.DEFINE_string("mode", "train", "The task to perform [train, evaluate]")
flags.DEFINE_integer("epoch", 1, "Epoch to train [25]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")

config = flags.FLAGS


def main(_):
    if config.mode == 'train':
        reader = TFReader(config.input_path, config.epoch, config.batch_size, [28*28], [10])
        cnn_model = CNN(reader)
        train(cnn_model, config)
    elif config.mode == 'evaluate':
        reader = TFReader(config.input_path, config.epoch, config.batch_size, [28 * 28], [10])
        cnn_model = CNN(reader, False, keep_prob=1.0)
        evaluate(cnn_model, config)
    else:
        pass


if __name__ == '__main__':
    tf.app.run()