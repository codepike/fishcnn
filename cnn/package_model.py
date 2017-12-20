import tensorflow as tf
import pickle
import argparse
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import builder as model_builder
from tensorflow.python.saved_model import tag_constants
import model

def build_model(config):
    builder = model_builder.SavedModelBuilder(config.model_dir)

    checkpoint_dir = config.checkpoint_dir
    cnn_model = model.CNN()
    with tf.Graph().as_default() as graph:
        x_batch = tf.placeholder(tf.float32, shape=[None, 64*64*3])
        logits = cnn_model.predict(x_batch)

        saver = tf.train.Saver()

        last_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        sess = tf.Session(graph=graph)

        saver.restore(sess, last_checkpoint)

        predict = tf.nn.softmax(logits)

        # build a model here
        x_info = tf.saved_model.utils.build_tensor_info(x_batch)
        y_info = tf.saved_model.utils.build_tensor_info(predict)

        pred_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'x': x_info},
            outputs={'price': y_info},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')


        builder.add_meta_graph_and_variables(
                sess,
                [tag_constants.SERVING],
                signature_def_map={'serving_default': pred_signature},
                legacy_init_op=legacy_init_op
        )

        builder.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--checkpoint_dir',
        help='The checkpoint file to build a model',
        required=True
    )

    parser.add_argument(
        '--model_dir',
        help='The model output dir',
        required=True
    )

    config = parser.parse_args()

    print(config)

    build_model(config)
