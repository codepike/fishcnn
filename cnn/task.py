import tensorflow as tf
import os
import traceback

class Train():
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
                'x': tf.FixedLenFeature([], tf.string),
                'y': tf.FixedLenFeature([], tf.string)
            }
        )

        x = tf.decode_raw(features['x'], tf.int64, name='x')
        y = tf.decode_raw(features['y'], tf.int64, name='y')

        x.set_shape([1])
        x = tf.cast(x, tf.float32)
        print('x', x)

        y.set_shape([1])
        y = tf.cast(y, tf.float32)

        capacity = batch_size*4
        print(batch_size)
        x_batch, y_batch = tf.train.shuffle_batch(
                [x,y],
                batch_size=batch_size,
                capacity=capacity,
                min_after_dequeue=1)

        return x_batch, y_batch

    def run(self, config):
        self.checkpoint_dir = config.checkpoint_dir

        with tf.Graph().as_default() as graph:
            x_batch, y_batch = self.read_data(config.dataset, config.epoch, config.batch_size)

            # self.model.build_graph(x_batch, y_batch)

            self.loss,w,b = self.model.build_graph(x_batch, y_batch)
            self.loss_mean = tf.reduce_mean(self.loss)
            tf.summary.scalar('loss', self.loss_mean)

            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.checkpoint_dir, graph)


            self.saver = tf.train.Saver()
            self.merged = tf.summary.merge_all()






            self.optimizer = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1)
            # self.optimizer = tf.train.AdamOptimizer(config.learning_rate)
            self.train = self.optimizer.minimize(self.loss)


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

                    # _, lo , summary, wv, bv = self.sess.run([self.train, self.loss_mean, self.merged,w,b])
                    _, lo, summary, wv, bv, xv, yv = self.sess.run([self.train, self.loss_mean, self.merged, w, b, x_batch, y_batch])
                    # print('loss', lo, wv, bv)
                    print('step: ', step, wv, bv, xv.shape, yv.shape)
                    self.writer.add_summary(summary, step)
                    # _, summary = self.sess.run([self.train, self.merged])
            except tf.errors.OutOfRangeError:
                # traceback.print_exc()
                print("finish training")
            finally:
                self.coord.request_stop()
                # self.saver.save(self.sess, self.checkpoint_dir)

            self.coord.join(threads)
            self.sess.close()


