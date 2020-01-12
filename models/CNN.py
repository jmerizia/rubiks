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
        #self.b3   = ResidBlock(256)
        #self.b4   = ResidBlock(256)
        self.drop = tfk.layers.Dropout(rate=dropout_rate)
        self.out = tfk.layers.Dense(1)

    @tf.function
    def call(self, x, training=False):
        x = self.fc(x)
        x = self.b1(x)
        x = self.drop(x, training=training)
        x = self.b2(x)
        #x = self.b3(x)
        #x = self.b4(x)
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
            self.optimizer     = tf.optimizers.Adam(self.learning_rate)
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
        
        #train_log_dir = 'logs/{}/train/'.format(run_name)
        #test_log_dir  = 'logs/{}/test/'.format(run_name)
        #cpu_log_dir   = 'logs/{}/cpu/'.format(run_name)
        #mem_log_dir   = 'logs/{}/mem/'.format(run_name)
        #tmp_log_dir   = 'logs/{}/tmp/'.format(run_name)
        #train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        #test_summary_writer  = tf.summary.create_file_writer(test_log_dir)
        #cpu_summary_writer   = tf.summary.create_file_writer(cpu_log_dir)
        #mem_summary_writer   = tf.summary.create_file_writer(mem_log_dir)
        #tmp_summary_writer   = tf.summary.create_file_writer(tmp_log_dir)

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
                    #histogram_freq=5,
                    #write_graph=True,
                    #write_images=True,
                    update_freq='epoch',
                    #embeddings_freq=5
                )
            ],
            validation_data=(test_x, test_y),
            shuffle=True
        )

        #num_examples = len(train_x)
        #test_x  = np.array(test_x)
        #test_y  = np.stack(test_y).reshape((-1, 1))

        #for epoch in range(1, self.epochs+1):
        #    print('=== Epoch ', epoch, '===')
        #    st = time.time()
        #    total_loss   = 0
        #    total_acc    = 0
        #    total_trials = 0
        #    step = 0


        #    # shuffle the data to improve regularization
        #    combined = list(zip(train_x, train_y))
        #    random.shuffle(combined)
        #    train_x = [x for x, y in combined]
        #    train_y = [y for x, y in combined]
        #    for i in range(0, num_examples, self.batch_size):
        #        step += 1
        #        batch_x = np.array(train_x[i:i+self.batch_size])
        #        batch_y = np.stack(train_y[i:i+self.batch_size]).reshape((-1, 1))

        #        # Run optimization op (backprop)
        #        self.run_optimization(batch_x, batch_y)

        #        if step % self.display_step == 0:
        #            # Calculate batch loss and accuracy
        #            pred = self.model(batch_x)
        #            loss = self.get_loss(pred, batch_y)
        #            acc  = self.get_acc(pred, batch_y)
        #            print("num_ex: {}, loss: {:0.4f}, acc: {:0.4f} (batch)" \
        #                    .format(i, loss, acc))
        #            total_loss   += loss
        #            total_acc    += acc
        #            total_trials += 1

        #    # evaluate test set loss / accuracy
        #    pred = self.model(test_x)
        #    test_loss = self.get_loss(pred, test_y)
        #    test_acc  = self.get_acc(pred, test_y)

        #    # log results for viewing in tensorboard
        #    with train_summary_writer.as_default():
        #        tf.summary.scalar('loss',     total_loss / total_trials, step=epoch)
        #        tf.summary.scalar('accuracy', total_acc  / total_trials, step=epoch)

        #    with test_summary_writer.as_default():
        #        tf.summary.scalar('loss',     test_loss, step=epoch)
        #        tf.summary.scalar('accuracy', test_acc,  step=epoch)

        #    with cpu_summary_writer.as_default():
        #        tf.summary.scalar('cpu', psutil.cpu_percent(), step=epoch)

        #    with mem_summary_writer.as_default():
        #        mem = psutil.virtual_memory()
        #        tf.summary.scalar('mem', mem.total - mem.available, step=epoch)

        #    with tmp_summary_writer.as_default():
        #        tmp = psutil.sensors_temperatures()
        #        tf.summary.scalar('tmp', tmp['k10temp'][0].current, step=epoch)

        #    en = time.time()
        #    print('elapsed time of epoch = {:0.4f} seconds'.format(en - st))


        print('Finished Training')

    def save(self, fname):
        self.model.save_weights(fname)

    def predict(self, x):
        return self.model(x)
