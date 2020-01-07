"""
Adapted from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network_raw.py
"""

from __future__ import absolute_import
import tensorflow as tf
from tensorflow.keras import Model, layers
import datetime
import os
import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Set memory to grow little-by-little as needed
# (because Tensorflow is shit at managing memory)
for device in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)


class ConvNet(Model):

    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = layers.Conv2D(64, kernel_size=5, activation=tf.nn.relu)
        self.maxpool1 = layers.MaxPool2D(2, strides=2)

        self.conv2 = layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu)
        self.maxpool2 = layers.MaxPool2D(2, strides=2)

        self.conv3 = layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu)
        self.maxpool3 = layers.MaxPool2D(2, strides=2)

        self.flatten = layers.Flatten()

        self.fc1 = layers.Dense(1000, activation=tf.nn.relu)
        self.fc2 = layers.Dense(1000, activation=tf.nn.relu)
        self.fc3 = layers.Dense(1000, activation=tf.nn.relu)
        self.fc4 = layers.Dense(1000, activation=tf.nn.relu)
        self.fc5 = layers.Dense(1000, activation=tf.nn.relu)

        self.dropout = layers.Dropout(rate=0.5)

        self.out = layers.Dense(1)

    @tf.function
    def call(self, x, is_training=False):
        #x = tf.reshape(x, [-1, 12, 12, 1])
        #x = self.conv1(x)
        #x = self.maxpool1(x)
        #x = self.conv2(x)
        #x = self.maxpool2(x)
        #x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.dropout(x, training=is_training)
        x = self.out(x)
        return x


class CNN:

    def __init__(self, fname=None):
        if not fname:
            self.conv_net = ConvNet()
            self.batch_size    = 128
            self.display_step  = 200
            self.learning_rate = 0.0005
            self.epochs        = 3000
            self.optimizer = tf.optimizers.Adam(self.learning_rate)
        else:
            self.conv_net = ConvNet()
            self.conv_net.load_weights(fname)

    def get_loss(self, x, y):
        loss = tf.nn.l2_loss(y - x)
        return tf.reduce_mean(loss)

    def get_acc(self, y_pred, y_true):
        diff = tf.math.abs(y_pred - y_true)
        correct = tf.less(diff, 0.14)
        return tf.reduce_mean(tf.cast(correct, tf.float32))

    def run_optimization(self, x, y):
        with tf.GradientTape() as g:
            pred = self.conv_net(x, is_training=True)
            loss = self.get_loss(pred, y)
            trainable_variables = self.conv_net.trainable_variables
            gradients = g.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    def profile_graph(self):
        writer = tf.summary.create_file_writer('logs/traces/')
        tf.summary.trace_on(graph=True, profiler=True)
        with writer.as_default():
            self.conv_net(np.random.rand(144))
            tf.summary.trace_export(
                'trace',
                step=0,
                profiler_outdir='logs/profiles')

    def train(self, train_x, train_y, test_x, test_y):

        num_examples = train_x.shape[0]

        # Setup tensorboard:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        train_log_dir = 'logs/' + current_time + '/train'
        test_log_dir = 'logs/' + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        for epoch in range(1, self.epochs+1):
            print('=== Epoch ', epoch, '===')
            total_loss   = 0
            total_acc    = 0
            total_trials = 0
            step = 0
            for i in range(0, num_examples, self.batch_size):
                step += 1
                batch_x = train_x[i:i+self.batch_size]
                batch_y = train_y[i:i+self.batch_size]

                # Run optimization op (backprop)
                self.run_optimization(batch_x, batch_y)

                if step % self.display_step == 0:
                    # Calculate batch loss and accuracy
                    pred = self.conv_net(batch_x)
                    loss = self.get_loss(pred, batch_y)
                    acc  = self.get_acc(pred, batch_y)
                    print("step: {}, loss: {}, acc: {}".format(step, loss, acc))
                    total_loss   += loss
                    total_acc    += acc
                    total_trials += 1

            # evaluate test set loss / accuracy
            pred = self.conv_net(test_x)
            test_loss = self.get_loss(pred, test_y)
            test_acc  = self.get_acc(pred, test_y)

            # log results for viewing in tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('loss',     total_loss / total_trials, step=epoch)
                tf.summary.scalar('accuracy', total_acc  / total_trials, step=epoch)

            with test_summary_writer.as_default():
                tf.summary.scalar('loss',     test_loss, step=epoch)
                tf.summary.scalar('accuracy', test_acc,  step=epoch)


        print('Finished Training')

    def save(self, fname):
        self.conv_net.save_weights(fname)

    def predict(self, x):
        return self.conv_net(x)
