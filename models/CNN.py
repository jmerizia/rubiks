"""
Adapted from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network_raw.py
"""

from __future__ import absolute_import
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class CNN:


    def __conv2d(self, x, W, b, strides=1):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)


    def __maxpool2d(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    def __conv_net(self, x, weights, biases, dropout):
        x = tf.reshape(x, shape=[-1, 12, 12, 1])
        conv1      = self.__conv2d(x, weights['wc1'], biases['bc1'])
        conv1_pool = self.__maxpool2d(conv1, k=2)
        conv2      = self.__conv2d(conv1_pool, weights['wc2'], biases['bc2'])
        conv2_pool = self.__maxpool2d(conv2, k=2)
        # flatten instead ?
        flat       = tf.reshape(conv2_pool, [-1, weights['wd1'].get_shape().as_list()[0]])
        # Dense linear instead ?
        fc1        = tf.add(tf.matmul(flat, weights['wd1']), biases['bd1'])
        activ1     = tf.nn.relu(fc1)
        drop       = tf.nn.dropout(activ1, dropout)
        # dense linear instead ?
        out        = tf.add(tf.matmul(drop, weights['out']), biases['out'])
        return out

    def __init__(self):
        """
        examples: a np array of N 2D examples with shape (N, height, width, 1).
        labels:   a np array of labels associated with each example.
        """

        self.display_step = 10
        self.learning_rate = 0.001
        self.batch_size = 256

        self.num_input = 144

        self.accuracy_epsilon = 20.0

        self.X = tf.placeholder(tf.float32, [None, self.num_input])
        self.Y = tf.placeholder(tf.float32, [None, 1])
        self.keep_prob = tf.placeholder(tf.float32)

        weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 3*3*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([3*3*64, 1024])),
            # 1024 inputs, 1 output (class prediction)
            'out': tf.Variable(tf.random_normal([1024, 1]))
        }

        biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([1]))
        }

        # Construct model
        self.prediction = self.__conv_net(self.X, weights, biases, self.keep_prob)

        # Define loss and optimizer
        self.loss_op = tf.nn.l2_loss(tf.math.subtract(self.prediction, self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss_op)

        # Evaluate model
        difference = tf.math.abs(tf.math.subtract(self.prediction, self.Y))
        correct_pred = tf.math.less(difference, tf.constant(self.accuracy_epsilon))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        self.init = tf.global_variables_initializer()

        # Start training
        self.sess = tf.Session()

    def __del__(self):
        if hasattr(self, 'sess'):
            self.sess.close()

    def train(self, examples, labels, epochs):

        num_examples = examples.shape[0]

        # Run the initializer
        self.sess.run(self.init)

        for epoch in range(1, epochs+1):
            print('=== Epoch ', epoch, '===')
            step = 0
            for i in range(0, num_examples, self.batch_size):
                step += 1
                batch_x = examples[i:i+self.batch_size]
                batch_y = labels[i:i+self.batch_size]
                # Run optimization op (backprop)
                self.sess.run(self.train_op, feed_dict={self.X: batch_x,
                                                        self.Y: batch_y,
                                                        self.keep_prob: 0.8})
                if step % self.display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = self.sess.run([self.loss_op, self.accuracy],
                                              feed_dict={self.X: batch_x,
                                                         self.Y: batch_y,
                                                         self.keep_prob: 1.0})
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))

        print('Finished Training')

    def predict(self, x):
        return self.sess.run(self.prediction, feed_dict={self.X: x, self.keep_prob: 1.0})
