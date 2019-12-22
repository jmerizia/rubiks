"""
Adapted from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network_raw.py
"""

from __future__ import absolute_import
import tensorflow as tf
from tensorflow.keras import Model, layers
import datetime

class ConvNet(Model):

    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = layers.Conv2D(32, kernel_size=5, activation=tf.nn.relu)
        self.maxpool1 = layers.MaxPool2D(2, strides=2)

        self.conv2 = layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu)
        self.maxpool2 = layers.MaxPool2D(2, strides=2)

        self.flatten = layers.Flatten()

        self.fc1 = layers.Dense(1024, activation=tf.nn.relu)
        self.fc2 = layers.Dense(512, activation=tf.nn.relu)
        self.fc3 = layers.Dense(512)
        self.dropout = layers.Dropout(rate=0.2)

        self.out = layers.Dense(1)

    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 12, 12, 1])
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.dropout(x, training=is_training)
        x = self.out(x)
        return x


class CNN:

    def __init__(self):
        self.conv_net = ConvNet()
        self.batch_size = 512
        self.display_step = 10
        self.learning_rate = 0.001
        self.optimizer = tf.optimizers.Adam(self.learning_rate)

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

    def train(self, train_x, train_y, test_x, test_y, epochs):

        num_examples = train_x.shape[0]

        # Setup tensorboard:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        for epoch in range(1, epochs+1):
            print('=== Epoch ', epoch, '===')
            total_loss = 0
            total_acc  = 0
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
                    total_loss  += loss
                    total_acc   += acc

            # evaluate test set loss / accuracy
            pred = self.conv_net(test_x)
            test_loss = self.get_loss(pred, test_y)
            test_acc  = self.get_acc(pred, test_y)

            # log results for viewing in tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('loss',     total_loss / step, step=epoch)
                tf.summary.scalar('accuracy', total_acc  / step, step=epoch)

            with test_summary_writer.as_default():
                tf.summary.scalar('loss',     test_loss, step=epoch)
                tf.summary.scalar('accuracy', test_acc,  step=epoch)

        print('Finished Training')

    def predict(self, x):
        return self.conv_net(x)
