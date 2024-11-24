import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.core.protobuf import rewriter_config_pb2
import audionet.params as params
from audionet.preprocess.input import waveform_to_examples
import audionet.preprocess.slim as slim

tf.disable_eager_execution()


class Daddy:
    def __init__(self, checkpoint_path):
        tf.Graph().as_default()
        self.sess = tf.Session(
            # config=tf.ConfigProto(
            #     graph_options=tf.GraphOptions(
            #         rewrite_options=rewriter_config_pb2.RewriterConfig(
            #             disable_meta_optimizer=True
            #         )
            #     )
            # )
        )

        slim.define_slim()
        slim.load_slim_checkpoint(self.sess, checkpoint_path)

    def forward(self, input_batch):
        features_tensor = self.sess.graph.get_tensor_by_name(params.INPUT_TENSOR_NAME)
        embedding_tensor = self.sess.graph.get_tensor_by_name(params.OUTPUT_TENSOR_NAME)
        [embedding_batch] = self.sess.run(
            [embedding_tensor], feed_dict={features_tensor: input_batch}
        )
        return embedding_batch

    def predict(self, input_audio):
        return self.forward(waveform_to_examples(input_audio, params.SAMPLE_RATE))
