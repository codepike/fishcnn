import tensorflow as tf
import os
import traceback
from scipy import misc
import numpy as np


def train(model, config):
    supervisor = tf.train.Supervisor(graph=model.graph, logdir=config.logdir, save_model_secs=60)

    with supervisor.managed_session() as sess:
        while not supervisor.should_stop():
            sess.run(model.train_op)


def evaluate(model, config):
    supervisor = tf.train.Supervisor(graph=model.graph, logdir=config.logdir, save_model_secs=0)

    with supervisor.managed_session() as sess:
        while not supervisor.should_stop():
            accuracy = sess.run([model.acc, model.acc_op])
            print("accuracy: ", accuracy)

        print("-----")
        print("accuracy: ", accuracy)

class TFReader:
    """A reader reads TFRecord files for training
    """
    def __init__(self,input_path, epoch, batch_size, x_shape, label_shape):
        self.input_path = input_path
        self.epoch = epoch
        self.batch_size = batch_size
        self.x_shape = x_shape
        self.label_shape = label_shape

    def read(self):
        if self.input_path.endswith('tfrecord'):
            filenames = [self.input_path]
        else:
            filenames = os.listdir(self.input_path)
            filenames = [name for name in filenames if name.endswith('.tfrecord')]
            filenames = [os.path.join(dataset, name) for name in filenames]

        # filename queues
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=self.epoch)

        reader = tf.TFRecordReader()
        key, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'x': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string)
            }
        )

        x = tf.decode_raw(features['x'], tf.uint8)

        x.set_shape(self.x_shape)
        x = tf.cast(x, tf.float32)

        label = tf.decode_raw(features['label'], tf.uint8)
        label.set_shape(self.label_shape)
        label = tf.cast(label, tf.float32)

        capacity = self.batch_size * 10

        x_batch, label_batch = tf.train.shuffle_batch(
            [x, label],
            batch_size=self.batch_size,
            capacity=capacity,
            min_after_dequeue=0)

        return x_batch, label_batch

class Task:
    def __init__(self):
        pass

    def read_data(self, dataset, epoch, batch_size, x_shape, label_shape):
        """Create a pipeline. It reads data from a TFRecord file if dataset is a
        file with .tfrecord. It reads all TFRecord files if the dataset is a dir.
        Args:
            dataset: a TFRecord file ending with .tfrecord or a directory containing
            TFRecord files
            epoch: the number of epoch reading data
            batch_size: the number of samples in a batch
        :return:a tuple of x batch and label batch
        """

        if dataset.endswith('tfrecord'):
            filenames = [dataset]
        else:
            filenames = os.listdir(dataset)
            filenames = [name for name in filenames if name.endswith('.tfrecord')]
            filenames = [os.path.join(dataset, name) for name in filenames]

        # filename queues
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=epoch)

        reader = tf.TFRecordReader()
        key, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features= {
                'x': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string)
            }
        )

        x = tf.decode_raw(features['x'], tf.uint8)

        x.set_shape(x_shape)
        x = tf.cast(x, tf.float32)

        label = tf.decode_raw(features['label'], tf.uint8)
        label.set_shape(label_shape)
        label = tf.cast(label, tf.float32)

        capacity = batch_size*1
        x_batch, label_batch = tf.train.shuffle_batch(
                [x,label],
                batch_size=batch_size,
                capacity=capacity,
                min_after_dequeue=0)

        return x_batch, label_batch

    def run(self, config):
        pass


class Train(Task):
    def __init__(self, model):
        self.model = model

    def run(self, config):
        checkpoint_dir = config.checkpoint_dir
        checkpoint = os.path.join(checkpoint_dir, 'model.ckpt')

        with tf.Graph().as_default() as graph:
            x_shape = [28*28*1]
            y_shape = [10]
            x_batch, y_batch = self.read_data(config.dataset,
                                              config.epoch,
                                              config.batch_size,
                                              x_shape,
                                              y_shape)

            # build graph
            loss = self.model.build_graph(x_batch, y_batch)
            tf.summary.scalar('loss', loss)

            writer = tf.summary.FileWriter(checkpoint_dir, graph)

            saver = tf.train.Saver(max_to_keep=5)
            merged = tf.summary.merge_all()

            optimizer = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1)
            train = optimizer.minimize(loss)

            with tf.Session(graph=graph) as sess:
                init_global = tf.global_variables_initializer()
                init_local = tf.local_variables_initializer()

                sess.run(init_global)
                sess.run(init_local)

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                try:
                    step = 0
                    while not coord.should_stop():
                        step += 1
                        _, summary= sess.run([train, merged])
                        if step % 100 == 0:
                            print(step)
                            writer.add_summary(summary, step)
                            saver.save(sess, checkpoint)
                except tf.errors.OutOfRangeError:
                    print("finish training")
                finally:
                    coord.request_stop()

                saver.save(sess, checkpoint)

                coord.join(threads)

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
                'data': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string)
            }
        )

        x = tf.decode_raw(features['data'], tf.uint8)
        y = tf.decode_raw(features['label'], tf.float64)

        x.set_shape([12288])
        x = tf.cast(x, tf.float32)
        print('x1 ',x)

        # x = tf.reshape(x, [-1, 28,28,1])
        print('x', x)

        y.set_shape([3])
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
        img = misc.imresize(img, (64,64))
        img = img.reshape(1,64*64*3).astype(np.float32)
        print(img.shape)
        self.checkpoint_dir = config.checkpoint_dir
        # checkpoint = os.path.join(self.checkpoint_dir, 'model.ckpt')

        with tf.Graph().as_default() as graph:
            x_batch = tf.placeholder(tf.float32, shape=[1, 12288])
            self.logits = self.model.predict(x_batch, 1.0)

            self.saver = tf.train.Saver()

            last_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
            print(last_checkpoint)

            self.sess = tf.Session(graph=graph)

            # init_global = tf.global_variables_initializer()
            # init_local = tf.local_variables_initializer()
            # self.sess.run(init_global)
            # self.sess.run(init_local)

            self.saver.restore(self.sess, last_checkpoint)

            # predict = tf.nn.softmax(self.logits)

            # predict = tf.reduce_mean(tf.arg_max(self.logits, 0))

            predict = tf.arg_max(self.logits, 1)
            prediction = self.sess.run([predict], feed_dict={x_batch: img})

            print("prediction: {}".format(prediction[0]))


class Evaluate(Task):
    def __init__(self, model):
        self.model = model

    def run(self, config):
        # get last check point
        checkpoint_dir = config.checkpoint_dir
        last_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

        with tf.Graph().as_default() as graph:
            x_shape = [28 * 28 * 1]

            x, label = self.read_data(config.dataset,
                                              1,
                                              config.batch_size,
                                              x_shape,
                                              [10])

            with tf.Session(graph=graph) as sess:
                init_global = tf.global_variables_initializer()
                init_local = tf.local_variables_initializer()
                sess.run(init_global)
                sess.run(init_local)

                logits = self.model.predict(x, 1.0)
                # restore last check point
                saver = tf.train.Saver()
                saver.restore(sess, last_checkpoint)

                prediction = tf.nn.softmax(logits)

                p1 = tf.argmax(prediction,1)

                correct_prediction = tf.equal(tf.argmax(label, 1), tf.argmax(prediction, 1))

                accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                try:
                    step = 0
                    while not coord.should_stop():
                        step += 1
                        # accuracy, p, l= sess.run([accuracy_op, prediction, label])
                        pp = sess.run([p1])
                        print("{},{}".format(step, pp[0][0]))
                except tf.errors.OutOfRangeError:
                    print("finish evaluating")
                finally:
                    coord.request_stop()

                coord.join(threads)

