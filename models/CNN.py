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

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Set memory to grow little-by-little as needed
# (because Tensorflow is shit at managing memory)
for device in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)

tf.keras.backend.set_floatx('float32')

class ResidFC(tfk.layers.Layer):

    def __init__(self, size, activation):
        super(ResidFC, self).__init__()
        self.size = size
        self.fc1 = tfk.layers.Dense(size, activation=activation)
        self.fc2 = tfk.layers.Dense(size)
        self.dropout = tfk.layers.Dropout(rate=0.5)

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

    def __init__(self, size, activation):
        super(ResidBlock, self).__init__()
        self.size = size
        self.res1 = ResidFC(size, activation)
        self.res2 = ResidFC(size, activation)
        self.res3 = ResidFC(size, activation)

    @tf.function
    def call(self, x, training=False):
        x = self.res1(x, training=training)
        x = self.res2(x, training=training)
        x = self.res3(x, training=training)
        return x


class ResidNet(tfk.Model):

    def __init__(self):
        super(ResidNet, self).__init__()

        self.fc = tfk.layers.Dense(256, activation=tf.nn.relu)
        self.b1 = ResidBlock(256, activation=tf.nn.relu)
        self.b2 = ResidBlock(256, activation=tf.nn.relu)
        self.b3 = ResidBlock(256, activation=tf.nn.relu)

        self.out1 = tfk.layers.Dense(1)
        self.out2 = tfk.layers.Dense(1)
        self.out3 = tfk.layers.Dense(1)

    @tf.function
    def call(self, x, training=False):
        x  = self.fc(x)
        x1 = self.b1(x, training=training)
        x2 = self.b2(x1, training=training)
        x3 = self.b3(x2, training=training)
        x3 = self.out3(x3)
        if training:
            x1 = self.out1(x1)
            x2 = self.out2(x2)
            return 0.1 * x1 + 0.1 * x2 + 0.8 * x3
        else:
            return x3



class CNN:

    def __init__(self, fname=None):
        if not fname:
            self.net = ResidNet()
            self.batch_size    = 128
            self.display_step  = 200
            self.learning_rate = 0.0005
            self.epochs        = 500
            self.optimizer = tf.optimizers.Adam(self.learning_rate)
        else:
            self.net = ConvNet()
            self.net.load_weights(fname)

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
            self.net(np.random.rand(144))
            tf.summary.trace_export(
                'trace',
                step=0,
                profiler_outdir='logs/profiles')

    def train(self, train_x, train_y, test_x, test_y):

        num_examples = len(train_x)
        test_x  = np.array(test_x)
        test_y  = np.stack(test_y).reshape((-1, 1))

        # Setup tensorboard:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        train_log_dir = 'logs/' + current_time + '/train'
        test_log_dir = 'logs/' + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

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

            en = time.time()
            print('elapsed time = {:0.4f} seconds'.format(en - st))


        print('Finished Training')

    def save(self, fname):
        self.net.save_weights(fname)

    def predict(self, x):
        return self.net(x)
