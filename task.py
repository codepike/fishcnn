import tensorflow as tf
import os
import traceback
from scipy import misc
import numpy as np

class Train():
    def __init__(self, model):
        self.model = model

    def read_data(self, dataset, epoch, batch_size):
        """Create a data pipeline

        Args:
            dataset: a list of file paths of TFRecord files
        :return:
        """

        if dataset.endswith('tfrecord'):
            filenames = [dataset]
        else:
            # dataset_dir = os.path.join('/home/jing/sandbox/fishcnn/data', dataset)
            dataset_dir = dataset
            filenames = os.listdir(dataset_dir)
            filenames = [name for name in filenames if name.endswith('.tfrecord')]
            filenames = [os.path.join(dataset_dir, name) for name in filenames]

        print(filenames)
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=epoch)

        reader = tf.TFRecordReader()
        key, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features= {
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string)
            }
        )

        x = tf.decode_raw(features['image_raw'], tf.uint8)
        x.set_shape([784])
        x = tf.cast(x, tf.float32)

        y = tf.decode_raw(features['label'], tf.float64)
        y.set_shape([10])
        y = tf.cast(y, tf.float32)

        capacity = batch_size*10
        x_batch, y_batch = tf.train.shuffle_batch(
                [x,y],
                batch_size=batch_size,
                capacity=capacity,
                min_after_dequeue=1)

        return x_batch, y_batch

    def run(self, config):
        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint = os.path.join(self.checkpoint_dir, 'model.ckpt')

        with tf.Graph().as_default() as graph:
            # read data
            x_batch, y_batch = self.read_data(config.dataset, config.epoch, config.batch_size)

            # build graph
            loss = self.model.build_graph(x_batch, y_batch)

            tf.summary.scalar('loss', loss)
            self.merged = tf.summary.merge_all()

            self.writer = tf.summary.FileWriter(self.checkpoint_dir, graph)

            self.saver = tf.train.Saver(max_to_keep=5)
            self.merged = tf.summary.merge_all()

            self.optimizer = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1)
            # self.optimizer = tf.train.AdamOptimizer(config.learning_rate)
            self.train = self.optimizer.minimize(loss)


            self.sess = tf.Session(graph=graph)
            init_global = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()
            self.sess.run(init_global)
            self.sess.run(init_local)
            self.coord = tf.train.Coordinator()

            threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

            # loaded, checkpoint_counter = self.load(self.checkpoint_dir)
            # if loaded:
            #     counter = checkpoint_counter
            #     print(" [*] Load succeeded")
            # else:
            #     print(" [!] Load failed...")

            try:
                step = 0
                while not self.coord.should_stop():
                    step += 1
                    _, summary= self.sess.run([self.train, self.merged])
                    if step % 100 == 0:
                        print(step)
                        self.writer.add_summary(summary, step)
                        self.saver.save(self.sess, self.checkpoint)
            except tf.errors.OutOfRangeError:
                print("finish training")
            finally:
                self.coord.request_stop()
                print(step)
                # self.writer.add_summary(summary, step)
                self.saver.save(self.sess, self.checkpoint)

            self.coord.join(threads)
            self.sess.close()


class Predict():
    def __init__(self, model):
        self.model = model

    def read_data(self, dataset, epoch, batch_size):
        """Create a data pipeline

        Args:
            dataset: a list of file paths of TFRecord files
        :return:
        """
        print("read")
        dataset_dir = os.path.join('/home/jing/sandbox/fishcnn/data', dataset)
        print(dataset_dir)
        filenames = os.listdir(dataset_dir)

        filenames = [name for name in filenames if name.endswith('.tfrecord')]
        filenames = [os.path.join(dataset_dir, name) for name in filenames]
        print('a',filenames)


        filename_queue = tf.train.string_input_producer(filenames, num_epochs=epoch)
        reader = tf.TFRecordReader()

        key, serialized_example = reader.read(filename_queue)
        print(key)
        features = tf.parse_single_example(
            serialized_example,
            features= {
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string)
            }
        )

        x = tf.decode_raw(features['image_raw'], tf.uint8)
        y = tf.decode_raw(features['label'], tf.float64)

        x.set_shape([784])
        x = tf.cast(x, tf.float32)
        print('x1 ',x)

        # x = tf.reshape(x, [-1, 28,28,1])
        print('x', x)

        y.set_shape([10])
        y = tf.cast(y, tf.float32)

        capacity = batch_size*4
        print(batch_size)
        x_batch, y_batch = tf.train.shuffle_batch(
                [x,y],
                batch_size=batch_size,
                capacity=capacity,
                min_after_dequeue=1)

        return x_batch, y_batch

    def run(self, config, sample):
        img = misc.imread(sample)
        img = img.reshape(1,784).astype(np.float32)
        print(img.shape)
        self.checkpoint_dir = config.checkpoint_dir
        # checkpoint = os.path.join(self.checkpoint_dir, 'model.ckpt')

        with tf.Graph().as_default() as graph:
            x_batch = tf.placeholder(tf.float32, shape=[1, 784])
            self.logits = self.model.predict(x_batch)

            self.saver = tf.train.Saver()

            last_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
            print(last_checkpoint)

            self.sess = tf.Session(graph=graph)

            # init_global = tf.global_variables_initializer()
            # init_local = tf.local_variables_initializer()
            # self.sess.run(init_global)
            # self.sess.run(init_local)

            self.saver.restore(self.sess, last_checkpoint)

            predict = tf.nn.softmax(self.logits)
            # predict = tf.arg_max(self.logits,1 )

            prediction = self.sess.run([predict], feed_dict={x_batch: img})

            print("prediction: {}".format(prediction))