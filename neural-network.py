# Import MNIST data
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow

# Hyper parameters
learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2



# Input
## Una imagen de la base de datos es de tamaño 28x28=784
x = tensorflow.placeholder("float", [None, 784])

# Output
## 0-9 dígitos => 10 clases
y = tensorflow.placeholder("float", [None, 10])

# Create a model

# Set model weights
## W = weights
W = tensorflow.Variable(tensorflow.zeros([784, 10]))
## b = biases
b = tensorflow.Variable(tensorflow.zeros([10]))

with tensorflow.name_scope("Wx_b") as scope:
    model = tensorflow.nn.softmax(tensorflow.matmul(x, W) + b)

w_h = tensorflow.histogram_summary("weigths", W)
b_h = tensorflow.histogram_summary("biases", b)
