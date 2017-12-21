import numpy as np
import tensorflow as tf
import os

flags = tf.app.flags

flags.DEFINE_string('output_path', '.', 'The path where output tfrecrod will be saved')
flags.DEFINE_float('split', '70', 'The percentage of train dataset')
flags.DEFINE_string('input_file', './train.csv', 'The input file to be converted into tf record file')

cfg = tf.app.flags.FLAGS

def read_file(filename):
    csv = np.genfromtxt(filename, delimiter=",")

    # remove header
    csv = csv[1:, :]

    # read labels from the first column
    labels = csv[:, 0:1]

    labels = labels.flatten()

    print(labels.shape)

    # read pixels from the second column and after
    data = csv[:, 1:]
    labels = labels.astype(np.int32)
    return labels, data

def create_tfrecrod(filename, labels, data):
    writer = tf.python_io.TFRecordWriter(filename)

    for i in range(labels.shape[0]):

        label = np.zeros(10)
        print(type(label))
        print(labels[i])
        print(type(labels[i]))
        label[labels[i]:labels[i]+1] = 1
        label = label.astype(np.uint8)

        x = data[i].astype(np.uint8)
        x = x.tostring()

        print(labels[i], label)

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'x': tf.train.Feature(bytes_list = tf.train.BytesList(value=[x])),
                    'label': tf.train.Feature(bytes_list = tf.train.BytesList(value=[label.tostring()])),
                },
            )
        )
        writer.write(example.SerializeToString())
    writer.close()

def create(cfg):
    labels, data = read_file(cfg.input_file)

    split = int(labels.shape[0]*cfg.split/100.0)

    create_tfrecrod(os.path.join(cfg.output_path, 'train.tfrecord'), labels[0:split], data[0:split, :])
    create_tfrecrod(os.path.join(cfg.output_path, 'validation.tfrecord'), labels[split:], data[split:,:])

def main(_):
    create(cfg)

if __name__ == '__main__':
    tf.app.run()