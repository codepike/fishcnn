import tensorflow as tf


def read_data(input_path, epoch, batch_size, x_shape, label_shape):
    """Read input data from TFRecord files. It the file if input_path is a TFRecord file;
    it reads from all TFRecord files if input_path is a directory.
    Args:
        input_path: a TFRecord file or a directory containing TFRecord files
        epoch: the number of epoch reading data
        batch_size: the number of samples in a batch
    :return:a tuple of x batch and label batch
    """

    if input_path.endswith('tfrecord'):
        filenames = [dataset]
    else:
        filenames = os.listdir(input_path)
        filenames = [name for name in filenames if name.endswith('.tfrecord')]
        filenames = [os.path.join(dataset, name) for name in filenames]

    # filename queues
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=epoch)

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

    x.set_shape(x_shape)
    x = tf.cast(x, tf.float32)

    label = tf.decode_raw(features['label'], tf.uint8)
    label.set_shape(label_shape)
    label = tf.cast(label, tf.float32)

    capacity = batch_size * 10

    x_batch, label_batch = tf.train.shuffle_batch(
        [x, label],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=0)

    return x_batch, label_batch