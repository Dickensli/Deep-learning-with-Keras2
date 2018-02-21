import tensorflow as tf
import keras.backend as K


def set_session(gpu_id):
    gpu_options = tf.GPUOptions(
        visible_device_list=str(gpu_id))
    config = tf.ConfigProto(gpu_options=gpu_options)
    session = tf.Session(config=config)
    K.set_session(session)
