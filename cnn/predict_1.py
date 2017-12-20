import tensorflow as tf
import pickle
import argparse
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import builder as model_builder
from tensorflow.python.saved_model import tag_constants
import model
from scipy import misc
import os

def predict(config):
    checkpoint_dir = config.checkpoint_dir
    cnn_model = model.CNN()
    with tf.Graph().as_default() as graph:
        x_batch = tf.placeholder(tf.float32, shape=[None, 64*64*3])
        logits = cnn_model.predict(x_batch)

        saver = tf.train.Saver()

        last_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

        sess = tf.Session(graph=graph)

        saver.restore(sess, last_checkpoint)

        predict = tf.argmax(tf.nn.softmax(logits), 1)


        if config.image is not None:
            image = misc.imread(config.image)
            image = misc.imresize(image, (64, 64, 3))
            image = image.reshape((1, 64 * 64 * 3))

            prediction = sess.run([predict], feed_dict={x_batch: image})

            print(prediction)

        if config.path is not None:
            dirnames = os.listdir(config.path)
            dirnames = sorted(dirnames)

            n = len(dirnames)
            index = 0
            for dirname in dirnames:
                path = os.path.join(config.path, dirname)
                filenames = os.listdir(path)
                for filename in filenames:
                    image = misc.imread(os.path.join(path, filename))
                    image = misc.imresize(image, (64, 64))
                    image = image.reshape(1,64*64*3)

                    label = sess.run([predict], feed_dict={x_batch: image})
                    # print(label, type(label))
                    print("{} : {}".format(dirname, dirnames[label[0][0]]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--checkpoint_dir',
        help='The checkpoint file to build a model',
        required=True
    )

    parser.add_argument(
        '--image',
        help='The image to predict',
        required=False
    )

    parser.add_argument(
        '--path',
        help='The directory containing images',
        required=False
    )

    config = parser.parse_args()

    print(config)

    predict(config)
