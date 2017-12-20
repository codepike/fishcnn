import tensorflow as tf
import numpy as np


def generate_data():
    filename = './linear.tfrecord'
    writer = tf.python_io.TFRecordWriter(filename)

    x = np.arange(10)
    x = x.astype(np.int64)
    y = np.arange(10)
    y = y.astype(np.int64)

    for i in range(10):
        print(x[i])
        # key, serialized_example = reader.read(filename_queue)
        #
        # key, serialized_example = reader.read(filename_queue)

        example = tf.train.Example(features=tf.train.Features(
            feature={
                'x': tf.train.Feature(int64_list=tf.train.Int64List(value=[x[i]])),
                'y': tf.train.Feature(int64_list=tf.train.Int64List(value=[y[i]])),
                # 'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x[i].tobytes()])),
                # 'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[y[i].tobytes()]))
                # # 'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(dataset.labels[index])]))
            }
        ))
        writer.write(example.SerializeToString())
    writer.close()

def main():
    generate_data()

if __name__ == '__main__':
    main()
