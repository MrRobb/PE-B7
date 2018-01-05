import math
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


# Parametros constantes
iterations = 5000
display_step = 10000
tf.set_random_seed(0)
sd = 0.1
img_size = 28
classification = 10
batch_size = 100


# Datos (input)
## mnist.test (10.000 imagenes + labels) y mnist.train (60.000 imagenes + labels)
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)


# Grafo

# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)

## [INPUT] 28 x 28 imagenes con un batch size "None" indeterminado
X = tf.placeholder(tf.float32, [None, img_size, img_size, 1])

# Hidden layers
K = 4       # convolutional 1
W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=sd))
B1 = tf.Variable(tf.truncated_normal([K], stddev=sd))

L = 8       # convolutional 2
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=sd))
B2 = tf.Variable(tf.truncated_normal([L], stddev=sd))

M = 16       # convolutional 3
W3 = tf.Variable(tf.truncated_normal([5, 5, L, M], stddev=sd))
B3 = tf.Variable(tf.truncated_normal([M], stddev=sd))

O = 32       # convolutional 3
W4 = tf.Variable(tf.truncated_normal([5, 5, M, O], stddev=sd))
B4 = tf.Variable(tf.truncated_normal([O], stddev=sd))

P = 64       # convolutional 3
W5 = tf.Variable(tf.truncated_normal([5, 5, O, P], stddev=sd))
B5 = tf.Variable(tf.truncated_normal([P], stddev=sd))

N = 200     # fully connected
W6 = tf.Variable(tf.truncated_normal([7 * 7 * P, N], stddev=sd))
B6 = tf.Variable(tf.truncated_normal([N], stddev=sd))

# Output layer
W7 = tf.Variable(tf.truncated_normal([N, classification], stddev=sd))
B7 = tf.Variable(tf.truncated_normal([classification], stddev=sd))

## [OUTPUT] 0,1,2...9 digitos, aqui iran las respuestas
Y_ = tf.placeholder(tf.float32, [None, classification])


# Modelo
# Convolutional
stride = 1  # 28x28 --> 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2	# 28x28 --> 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 1	# 14x14 --> 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)
stride = 2	# 7x7 --> 7x7
Y4 = tf.nn.relu(tf.nn.conv2d(Y3, W4, strides=[1, stride, stride, 1], padding='SAME') + B4)
stride = 1	# 7x7 --> 7x7
Y5 = tf.nn.relu(tf.nn.conv2d(Y4, W5, strides=[1, stride, stride, 1], padding='SAME') + B5)
# Fully connected
Y_reshape = tf.reshape(Y5, shape=[-1, 7 * 7 * P])   # necesario para pasar de conv --> fully connected
Y6 = tf.nn.relu(tf.matmul(Y_reshape, W6) + B6)
Y_intermediate = tf.matmul(Y6, W7) + B7             # necesario para cross entropy
Y = tf.nn.softmax(Y_intermediate)


# Error
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y_intermediate, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*batch_size

# Calcular precision (0..1)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Entrenamiento
lr = tf.placeholder(tf.float32)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)


# Init
init = tf.global_variables_initializer()

train_loss = list()
train_acc = list()
test_loss = list()
test_acc = list()

# Training loop
with tf.Session() as sess:
    sess.run(init)

    for i in range(iterations + 1):
        batch_X, batch_Y = mnist.train.next_batch(batch_size)

        max_learning_rate = 0.003
        min_learning_rate = 0.0001
        decay_speed = 2000.0
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

        if i % display_step == 0:
            # training
            train_acc, train_loss = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y, pkeep: 1.0})

            # testing
            test_acc, test_loss = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_:mnist.test.labels, pkeep: 1.0})

            # print
            print("{}\t{}\t{}\t{}\t{}".format(i,train_acc,train_loss,test_acc,test_loss))

        sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, lr: learning_rate, pkeep: 0.75})
