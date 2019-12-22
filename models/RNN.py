import tensorflow as tf
from tensorflow.contrib import rnn
import random

class RNN:
    """
    Recurrent neural network taken from Tensorflow examples.
    """

    # Training Parameters
    learning_rate = 0.001
    batch_size = 128
    display_step = 200

    # Network Parameters
    num_input = 28
    num_hidden = 128
    num_classes = 10

    # tf Graph input
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    def RNN(self, x, weights, biases):
        num_hidden = 128
        lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    logits = RNN(X, weights, biases)
    prediction = tf.nn.softmax(logits)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def __init__(self, examples):

        # Initialize the session and train:
        init = tf.global_variable_initializer()
        self.sess = tf.Session()
        sess.run(init)
        random.shuffle(examples)
        for i in range(0, len(examples), batch_size):
            batch_x = []
            batch_y = []
            for state, distance in examples[i:i+batch_size]:
                batch_y.append(example
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

    def predict(examples)
        return self.sess.run(prediction, feed_dict={X: examples})
