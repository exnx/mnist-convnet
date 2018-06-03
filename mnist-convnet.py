import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# the mnist data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_classes = 10      # output classes
batch_size = 128       # size of the batches for reading in data

# placeholder vales for the data, form, height x width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')  # actual labels to be passed at call

# drop out parameters
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    # move 1 pixel at a time. second and third entries refer to the 2d image
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    # move a 2x2 window, 2 steps at a time. second and third entries refer to the 2d image
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def conv_net(x):

    # define the layers in cnn

    # list positions:  5 x 5 convolution, 1 channel, produce 32 output/features
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
                'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
                'W_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),  #  7x7 input, 64 channels, output 1024
                'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'b_out': tf.Variable(tf.random_normal([n_classes]))}

    # -1 refers to unknown number of images, 2nd + 3rd image dim., 4th is num of channels
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # create the layers.  (conv2d,relu,maxpool)
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    # flattening result
    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

    # optional drop out rate here
    fc = tf.nn.dropout(fc, keep_rate)

    # final output layer
    output = tf.matmul(fc, weights['out']) + biases['b_out']

    return output


def train_neural_network(x):
    prediction = conv_net(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # optional, can put learning rate, default is 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycles of feed forward + backprop
    num_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # initialize all variables, begins the session

        # training network using data
        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)   # grabs batches of x (image input) and y (labels) data
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch + 1, 'completed out of', num_epochs, 'loss: ', epoch_loss)

        # after optimizing weights, run test data through model, and compare accuracy
            # argmax() returns 1 for highest value in vector, for each sample.
            # equal() will check if values are same (1 or 0 are the same)
        correct = tf.equal(tf.argmax(prediction, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))  # convert to float value
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))  # evaluate with the data

train_neural_network(x)


