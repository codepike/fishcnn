import tensorflow as tf
import os


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


class TFReader:
    """A reader reads TFRecord files
    Args:

    """
    def __init__(self,data_pah, epoch, batch_size, x_shape, label_shape):
        self.data_path = data_pah
        self.epoch = epoch
        self.batch_size = batch_size
        self.x_shape = x_shape
        self.label_shape = label_shape

    def read(self):
        if self.data_path.endswith('tfrecord'):
            filenames = [self.data_path]
        else:
            filenames = os.listdir(self.data_path)
            filenames = [name for name in filenames if name.endswith('.tfrecord')]
            filenames = [os.path.join(self.data_path, name) for name in filenames]

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