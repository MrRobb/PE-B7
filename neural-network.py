# Importar MNIST data
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow

# Hyper parámetros
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



# Crear modelo

# Poner weights del model
## W = weights
W = tensorflow.Variable(tensorflow.zeros([784, 10]))
## b = biases
b = tensorflow.Variable(tensorflow.zeros([10]))

with tensorflow.name_scope("Wx_b") as scope:
    model = tensorflow.nn.softmax(tensorflow.matmul(x, W) + b)

# Añadir summary operaciones para obtener los datos
w_h = tensorflow.histogram_summary("weigths", W)
b_h = tensorflow.histogram_summary("biases", b)

with tensorflow.name_scope("cost_function") as scope:
    cost_function = -tensorflow.reduce_sum(y * tensorflow.log(model))
    tensorflow.scalar_summary("cost_function", cost_function)

with tensorflow.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

init = tf.initialize_all_variables()

merged_summary_op = tf.merge_all_summaries()
