"""
tensorflow backend (https://github.com/tensorflow/tensorflow)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
import tensorflow.contrib.tensorrt as trt

import backend


class BackendTensorflowRT(backend.Backend):
    def __init__(self):
        super(BackendTensorflowRT, self).__init__()

    def set_extra_params (self, params):
        self.params = params

    def version(self):
        return tf.__version__ + "/" + tf.__git_version__

    def name(self):
        return "tensorflowRT"

    def image_format(self):
        # By default tensorflow uses NHWC (and the cpu implementation only does NHWC)
        return "NHWC"

    def load(self, model_path, inputs=None, outputs=None):
        # there is no input/output meta data i the graph so it need to come from config.
        if not inputs:
            raise ValueError("BackendTensorflow needs inputs")
        if not outputs:
            raise ValueError("BackendTensorflow needs outputs")
        self.outputs = outputs
        self.inputs = inputs

        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.allocator_type = 'BFC'
 #       tf_config.gpu_options.per_process_gpu_memory_fraction = mem_percent / 100.0 
 #       if num_processors > 0: 
 #           tf_config.device_count["CPU"] = num_processors
        
        
        tf.compat.v1.Graph().as_default()
        session = tf.compat.v1.Session(config=tf_config)
        graph_def = tf.compat.v1.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        trt_graph = trt.create_inference_graph(
              input_graph_def=graph_def,
              outputs=['num_detections:0','detection_boxes:0','detection_scores:0','detection_classes:0'],
              max_batch_size=self.params["BATCH_SIZE"],
              max_workspace_size_bytes=4000000000,
              is_dynamic_op=True if self.params["TENSORRT_DYNAMIC"]==1 else False,
              precision_mode=self.params["TENSORRT_PRECISION"]
              )
        tf.import_graph_def(
              trt_graph,
              return_elements=['num_detections:0','detection_boxes:0','detection_scores:0','detection_classes:0'])
        # TODO: support checkpoint and saved_model formats?
#        graph_def = graph_pb2.GraphDef()
#        with open(model_path, "rb") as f:
#            graph_def.ParseFromString(f.read())
#        g = tf.import_graph_def(graph_def, name='')
        self.sess = session #tf.Session(graph=g)
        return self

    def predict(self, feed):
        return self.sess.run(self.outputs, feed_dict=feed)
