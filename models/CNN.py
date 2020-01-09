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
import psutil

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Set memory to grow little-by-little as needed
# (because Tensorflow is shit at managing memory)
for device in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)

tf.keras.backend.set_floatx('float32')

class ResidFC(tfk.layers.Layer):

    def __init__(self, size, activation, dropout_rate):
        super(ResidFC, self).__init__()
        self.size = size
        self.fc1 = tfk.layers.Dense(size, activation=activation)
        self.fc2 = tfk.layers.Dense(size)
        self.dropout = tfk.layers.Dropout(rate=dropout_rate)

    @tf.function
    def call(self, x, training=False):
        orig = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        x = tf.add(orig, x)
        x = tf.nn.relu(x)
        return x


class ResidBlock(tfk.layers.Layer):

    def __init__(self, size, activation, dropout_rate):
        super(ResidBlock, self).__init__()
        self.size = size
        self.res1 = ResidFC(size, activation=activation, dropout_rate=dropout_rate)
        self.res2 = ResidFC(size, activation=activation, dropout_rate=dropout_rate)
        self.res3 = ResidFC(size, activation=activation, dropout_rate=dropout_rate)

    @tf.function
    def call(self, x, training=False):
        x = self.res1(x, training=training)
        x = self.res2(x, training=training)
        x = self.res3(x, training=training)
        return x


class ResidNet(tfk.Model):

    def __init__(self, dropout_rate):
        super(ResidNet, self).__init__()

        self.fc = tfk.layers.Dense(256, activation=tf.nn.relu)
        self.b1 = ResidBlock(256, activation=tf.nn.relu, dropout_rate=dropout_rate)
        self.b2 = ResidBlock(256, activation=tf.nn.relu, dropout_rate=dropout_rate)
        self.b3 = ResidBlock(256, activation=tf.nn.relu, dropout_rate=dropout_rate)

        self.out = tfk.layers.Dense(1)

    @tf.function
    def call(self, x, training=False):
        x = self.fc(x)
        x = self.b1(x, training=training)
        x = self.b2(x, training=training)
        x = self.b3(x, training=training)
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
            self.net           = ResidNet(dropout_rate)
            self.model_name    = model_name
            self.batch_size    = batch_size
            self.display_step  = display_step
            self.learning_rate = learning_rate
            self.epochs        = epochs
            self.optimizer     = tf.optimizers.Adam(self.learning_rate)
        else:
            # we must load an existing one
            self.net = ResidNet(dropout_rate)
            self.net.load_weights(model_fname)

    def get_loss(self, x, y):
        loss = tf.nn.l2_loss(y - x)
        return tf.reduce_mean(loss)

    def get_acc(self, y_pred, y_true):
        diff = tf.math.abs(y_pred - y_true)
        correct = tf.less(diff, 0.14)
        return tf.reduce_mean(tf.cast(correct, tf.float32))

    def run_optimization(self, x, y):
        with tf.GradientTape() as g:
            pred = self.net(x, training=True)
            loss = self.get_loss(pred, y)
            trainable_variables = self.net.trainable_variables
            gradients = g.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    def profile_graph(self):
        writer = tf.summary.create_file_writer('logs/traces/')
        tf.summary.trace_on(graph=True, profiler=True)
        with writer.as_default():
            self.net(np.random.rand(1, 144))
            tf.summary.trace_export(
                self.model_name,
                step=0,
                profiler_outdir='logs/profiles')

    def train(self, train_x, train_y, test_x, test_y):

        num_examples = len(train_x)
        test_x  = np.array(test_x)
        test_y  = np.stack(test_y).reshape((-1, 1))

        # Setup tensorboard:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        train_log_dir = 'logs/{}-{}/train'.format(current_time, self.model_name)
        test_log_dir  = 'logs/{}-{}/test'.format(current_time, self.model_name)
        cpu_log_dir   = 'logs/{}-{}/cpu'.format(current_time, self.model_name)
        mem_log_dir   = 'logs/{}-{}/mem'.format(current_time, self.model_name)
        tmp_log_dir   = 'logs/{}-{}/tmp'.format(current_time, self.model_name)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer  = tf.summary.create_file_writer(test_log_dir)
        cpu_summary_writer   = tf.summary.create_file_writer(cpu_log_dir)
        mem_summary_writer   = tf.summary.create_file_writer(mem_log_dir)
        tmp_summary_writer   = tf.summary.create_file_writer(tmp_log_dir)

        for epoch in range(1, self.epochs+1):
            print('=== Epoch ', epoch, '===')
            st = time.time()
            total_loss   = 0
            total_acc    = 0
            total_trials = 0
            step = 0

            # shuffle the data to improve regularization
            combined = list(zip(train_x, train_y))
            random.shuffle(combined)
            train_x = [x for x, y in combined]
            train_y = [y for x, y in combined]
            for i in range(0, num_examples, self.batch_size):
                step += 1
                batch_x = np.array(train_x[i:i+self.batch_size])
                batch_y = np.stack(train_y[i:i+self.batch_size]).reshape((-1, 1))

                # Run optimization op (backprop)
                self.run_optimization(batch_x, batch_y)

                if step % self.display_step == 0:
                    # Calculate batch loss and accuracy
                    pred = self.net(batch_x)
                    loss = self.get_loss(pred, batch_y)
                    acc  = self.get_acc(pred, batch_y)
                    print("num_ex: {}, loss: {:0.4f}, acc: {:0.4f} (batch)" \
                            .format(i, loss, acc))
                    total_loss   += loss
                    total_acc    += acc
                    total_trials += 1

            # evaluate test set loss / accuracy
            pred = self.net(test_x)
            test_loss = self.get_loss(pred, test_y)
            test_acc  = self.get_acc(pred, test_y)

            # log results for viewing in tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('loss',     total_loss / total_trials, step=epoch)
                tf.summary.scalar('accuracy', total_acc  / total_trials, step=epoch)

            with test_summary_writer.as_default():
                tf.summary.scalar('loss',     test_loss, step=epoch)
                tf.summary.scalar('accuracy', test_acc,  step=epoch)

            with cpu_summary_writer.as_default():
                tf.summary.scalar('cpu', psutil.cpu_percent(), step=epoch)

            with mem_summary_writer.as_default():
                mem = psutil.virtual_memory()
                tf.summary.scalar('mem', mem.total - mem.available, step=epoch)

            with tmp_summary_writer.as_default():
                tmp = psutil.sensors_temperatures()
                tf.summary.scalar('tmp', tmp['k10temp'][0].current, step=epoch)

            en = time.time()
            print('elapsed time of epoch = {:0.4f} seconds'.format(en - st))


        print('Finished Training')

    def save(self, fname):
        self.net.save_weights(fname)

    def predict(self, x):
        return self.net(x)
