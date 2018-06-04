import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# the mnist data (included in TensorFlow)
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# model parameters
n_classes = 10      # output classes (the 10 numbers)
batch_size = 128       # size of each batch when reading in data (can't do all at once)

# placeholder vales for the data, form, height x width (will be assigned when running TF)
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')  # actual labels to be passed at call

# drop out parameters
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    '''
    Function to convolve over an images.  Moves 1 pixel at a time. Second and 
    third entries in output refer to the 2d image.
    
    input params
    x: tensor, image data (from the batch)
    W: tensor, Weights (as defined in the convnet layers)
    
    return: tensor, the convolved sum
    '''
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    '''
    Function to max pool over the data. Move a 2x2 window, 2 steps at a time. 
    Second and third entries in output refer to the 2d image.
    
    input params
    x: tensor, image data (from the batch)
    
    return: tensor, the maxpool
    '''
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convnet(x):
    '''
    This function defines the layers in the convnet.
    
    input params
    x:  tensor, image data (from the batch)
    
    return: tensor, values for final each 10 classes (higher is more likely that number)
    
    '''
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

    # create the conv layers.  (conv2d,relu,maxpool)
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    # create fully connected layer by flattening result and apply relu
    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

    # optional drop out rate here
    fc = tf.nn.dropout(fc, keep_rate)

    # final output layer values for each image in batch, with 10 classes
    output = tf.matmul(fc, weights['out']) + biases['b_out']

    return output

def train_convnet(x):
    '''
    This function runs the training of the convnet. It predicts the class values
    by calling convnet(), applies softmax to the prediction and finds the cost 
    associated with that batch against the known labels.  It optimizes using Adam, 
    instead of SGD.
    
    input params 
    x: tensor, image from the batch
    
    return: tensor, output, the prediction values for each of the 10 classes
    
    '''
    prediction = convnet(x)
    # applies softmax on prediction values, and minimizes cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # optional, can put learning rate, default is 0.001.  Adam is a more complex
    # backprop method than SGD
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycles of feed forward + backprop = epoch
    num_epochs = 10

    # running TensorFlow, this is very idiomatic of TF
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # initialize all variables, begins the session

        # training network using data
        for epoch in range(num_epochs):
            epoch_loss = 0  # intialize loss for this epoch cycle
            # iterate through mnist training examples, one batch size / time
            for _ in range(int(mnist.train.num_examples/batch_size)):
                # grabs batches of x (image input) and y (labels) data
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)   
                # evaluates cost for this single sample, while feeding x and y values to TF
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c  # accumm the loss for the epoch
            # print out progress
            print('Epoch', epoch + 1, 'completed out of', num_epochs, 'loss: ', epoch_loss)

        # after optimizing weights, run test data through model, and compare accuracy
            # argmax() returns 1 for highest value in vector, for each sample.
            # equal() will check if values are same (1 or 0 are the same)
        correct = tf.equal(tf.argmax(prediction, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))  # convert to float value
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))  # evaluate with the data

# run convnet (x gets assigned by TF when the session runs)
train_convnet(x)
