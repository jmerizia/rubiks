"""
Adapted from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network_raw.py
"""

from __future__ import absolute_import
import tensorflow as tf
import tensorflow.keras as tfk
import datetime
import os
import numpy as np
import random
import time
#import psutil

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# Set memory to grow little-by-little as needed
# (because Tensorflow is shit at managing memory)
for device in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)

tf.keras.backend.set_floatx('float32')

class ResidFC(tfk.layers.Layer):

    def __init__(self, size):
        super(ResidFC, self).__init__()
        self.size = size
        self.fc1 = tfk.layers.Dense(size, activation=tf.nn.relu)
        self.fc2 = tfk.layers.Dense(size)

    @tf.function
    def call(self, x):
        orig = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = tf.add(orig, x)
        x = tf.nn.relu(x)
        return x


class ResidBlock(tfk.layers.Layer):

    def __init__(self, size):
        super(ResidBlock, self).__init__()
        self.size = size
        self.res1 = ResidFC(size)
        self.res2 = ResidFC(size)
        self.res3 = ResidFC(size)

    @tf.function
    def call(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x


class ResidNet(tfk.Model):

    def __init__(self, dropout_rate):
        super(ResidNet, self).__init__()
        self.fc   = tfk.layers.Dense(256, activation=tf.nn.relu)
        self.b1   = ResidBlock(256)
        self.b2   = ResidBlock(256)
        self.b3   = ResidBlock(256)
        self.b4   = ResidBlock(256)
        self.out = tfk.layers.Dense(1)

    @tf.function
    def call(self, x, training=False):
        x = self.fc(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.out(x)
        return x

class CNN:

    def __init__(self,
                 model_fname=None,
                 model_name='my_model',
                 learning_rate=0.0001,
                 batch_size=128,
                 display_step=200,
                 dropout_rate=0.5,
                 epochs=200):
        if not model_fname:
            # we must train a new one
            self.model         = ResidNet(dropout_rate)
            self.model_name    = model_name
            self.batch_size    = batch_size
            self.display_step  = display_step
            self.learning_rate = learning_rate
            self.epochs        = epochs
            self.lr_decay      = learning_rate / epochs
            self.optimizer     = tf.optimizers.Adam(
                    learning_rate=self.learning_rate,
                    decay=self.lr_decay
                    )
        else:
            # we must load an existing one
            self.model = ResidNet(dropout_rate)
            self.model.load_weights(model_fname)

    @tf.function
    def get_loss(self, x, y):
        loss = tf.nn.l2_loss(y - x)
        return tf.reduce_mean(loss)

    @tf.function
    def get_acc(self, y_pred, y_true):
        diff = tf.math.abs(y_pred - y_true)
        correct = tf.less(diff, 0.14)
        return tf.reduce_mean(tf.cast(correct, tf.float32))

    def train(self, train_x, train_y, test_x, test_y):

        # Compile the model:
        print('Begin compile()')
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.get_loss,
            metrics=[self.get_acc]
        )

        # Setup tensorboard:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = '{}-{}'.format(current_time, self.model_name)
        log_dir = 'logs/{}/'.format(run_name)

        print('Begin fit()')
        self.model.fit(
            x=train_x,
            y=train_y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=2,
            callbacks=[
                tfk.callbacks.TensorBoard(
                    log_dir=log_dir,
                    # temporary workaround (https://github.com/tensorflow/tensorboard/issues/2084):
                    profile_batch=0, 
                    update_freq='epoch',
                )
            ],
            validation_data=(test_x, test_y),
            shuffle=True
        )

        print('Finished Training')

    def save(self, fname):
        self.model.save_weights(fname)

    def predict(self, x):
        return self.model(x)
