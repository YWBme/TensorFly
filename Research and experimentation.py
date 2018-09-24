# Research and experimentation


# ----------------------------------------------------------------------------------------------------------------------
# Eager execution basics  |  TensorFlow
# ----------------------------------------------------------------------------------------------------------------------
# https://www.tensorflow.org/tutorials/eager/eager_basics

# Import TensorFlow
import tensorflow as tf
tf.enable_eager_execution()
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.encode_base64("hello world"))
# Operator overloading is also supported
print(tf.square(2) + tf.square(3))

x = tf.matmul([[1]], [[2, 3]])
print(x.shape)
print(x.dtype)


import numpy as np
ndarray = np.ones([3, 3])
print(ndarray)
print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)
print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))
print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())


# GPU acceleration
x = tf.random_uniform([3, 3])
print("Is there a GPU available: "),
print(tf.test.is_gpu_available())
print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))




def time_matmul(x):
    %timeit tf.matmul(x, x)
# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
    x = tf.random_uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)
# Force execution on GPU #0 if available
if tf.test.is_gpu_available():
     with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
          x = tf.random_uniform([1000, 1000])
          assert x.device.endswith("GPU:0")
          time_matmul(x)


# Datasets

ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
# Create a CSV file
import tempfile
_, filename = tempfile.mkstemp()
with open(filename, 'w') as f:
    f.write("""Line 1
Line 2
Line 3
  """)
ds_file = tf.data.TextLineDataset(filename)

ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)


# Iterate
print('Elements of ds_tensors:')
for x in ds_tensors:
    print(x)
print('\nElements in ds_file:')
for x in ds_file:
    print(x)



# ----------------------------------------------------------------------------------------------------------------------
# Automatic differentiation and gradient tape  |  TensorFlow
# ----------------------------------------------------------------------------------------------------------------------
# https://www.tensorflow.org/tutorials/eager/automatic_differentiation

# Setup
import tensorflow as tf
tf.enable_eager_execution()
tfe = tf.contrib.eager # Shorthand for some symbols


# Derivatives of a function
from math import pi
def f(x):
    return tf.square(tf.sin(x))
assert f(pi/2).numpy() == 1.0
# grad_f will return a list of derivatives of f
# with respect to its arguments. Since f() has a single argument,
# grad_f will return a list with a single element.
grad_f = tfe.gradients_function(f)
assert tf.abs(grad_f(pi/2)[0]).numpy() < 1e-7


# Higher-order gradients
def f(x):
    return tf.square(tf.sin(x))
def grad(f):
    return lambda x: tfe.gradients_function(f)(x)[0]
x = tf.lin_space(-2*pi, 2*pi, 100)  # 100 points between -2π and +2π
import matplotlib.pyplot as plt
plt.plot(x, f(x), label="f")
plt.plot(x, grad(f)(x), label="first derivative")
plt.plot(x, grad(grad(f))(x), label="second derivative")
plt.plot(x, grad(grad(grad(f)))(x), label="third derivative")
plt.legend()
plt.show()


# Gradient tapes
def f(x, y):
    output = 1
    for i in range(int(y)):
        output = tf.multiply(output, x)
    return output

f(1,1)
f(1,2)
f(2,2)
f(4,1)
f(4,4).numpy() == 4**4
f(5,4).numpy() == 5**4
f(23,12).numpy() == 23**12
f(8.0,88).numpy() == 8.0**80
f(7,9).numpy() == 7**9


def g(x, y):
    return tfe.gradients_function(f)(x, y)[0]

assert f(3.0,2).numpy() == 9.0
assert g(3.0,2).numpy() == 6.0
assert f(4.0,3).numpy() == 64.0
assert g(4.0,3).numpy() == 48.0



x = tf.ones((2,2))
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y,y)

dz_dy = t.gradient(z, y)
assert dz_dy.numpy() == 8.0

dz_dx = t.gradient(z, x)
dz_dx
for i in [0,1]:
    for j in [0,1]:
        assert dz_dx[i][j].numpy() == 8.0



# Higher-order gradients
x = tf.constant(1.0)  # Convert the Python 1.0 to a Tensor object
with tf.GradientTape() as t:
    with tf.GradientTape() as t2:
        t2.watch(x)
        y = x * x * x
  # Compute the gradient inside the 't' context manager
  # which means the gradient computation is differentiable as well.
        dy_dx = t2.gradient(y, x)
        d2y_dx2 = t.gradient(dy_dx, x)
assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0








# ----------------------------------------------------------------------------------------------------------------------
# Custom training: basics
# ----------------------------------------------------------------------------------------------------------------------
# https://www.tensorflow.org/tutorials/eager/custom_training


# Setup

import tensorflow as tf
tfe = tf.contrib.eager
tf.enable_eager_execution()


# Variables
x = tf.zeros([10,10])
x
x += 2
x
print(x)

v = tfe.Variable(1.0)
v
assert v.numpy() == 1.0

v.assign(3.0)
v
assert v.numpy() == 3.0

v.assign(tf.square(v))
assert v.numpy() == 9.0
v


# Example: Fitting a linear model

# Define the model
class Model(object):
    def __init__(self):
        self.W = tfe.Variable(5.0)
        self.b = tfe.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b

model = Model()
assert model(3.0).numpy() == 15.0

# Define a loss function
def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))

# Obtain training data
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random_normal(shape=[NUM_EXAMPLES])
noise = tf.random_normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

inputs
noise

import matplotlib.pyplot as plt
plt.scatter(inputs, outputs, c='b')
plt.scatter(inputs, model(inputs), c='r')
plt.show()

print('Current loss:')
print(loss(model(inputs), outputs).numpy())


# Define a training loop

def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)

    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)

model = Model()

Ws, bs = [], []

epochs = range(20)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs), outputs)
    train(model, inputs, outputs, learning_rate=0.1)
    print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %(epoch, Ws[-1], bs[-1], current_loss))


plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')

plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')

plt.legend(['W', 'b', 'true_W', 'true_b'])
plt.show()


# plt.plot(epochs, Ws, 'r', label='W')
# plt.plot(epochs, bs, 'b', label='b')
#
# plt.plot([TRUE_W] * len(epochs), 'r--', label = 'True_W')
# plt.plot([TRUE_b] * len(epochs), 'b--', label = 'True_b')
#
# plt.legend()
# plt.show()




# ----------------------------------------------------------------------------------------------------------------------
# Custom layers  |  TensorFlow
# ----------------------------------------------------------------------------------------------------------------------
# https://www.tensorflow.org/tutorials/eager/custom_layers


import tensorflow as tf
tfe = tf.contrib.eager
tf.enable_eager_execution()

# Layers: common sets of useful operations

# In the tf.keras.layers package, layers are objects. To construct a layer,
# simply construct the object. Most layers take as a first argument the number
# of output dimensions / channels.
layer = tf.keras.layers.Dense(100)
# The number of input dimensions is often unnecessary, as it can be inferred
# the first time the layer is used, but it can be provided if you want to
# specify it manually, which is useful in some complex models.
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))

# To use a layer, simply call it.
layer(tf.zeros([10, 5]))


# Layers have many useful methods. For example, you can inspect all variables
# in a layer by calling layer.variables. In this case a fully-connected layer
# will have variables for weights and biases.
layer.variables

# The variables are also accessible through nice accessors
layer.kernel, layer.bias


class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
                                        shape=[input_shape[-1].value,
                                               self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)


layer = MyDenseLayer(10)
print(layer(tf.zeros([10, 5])))
print(layer.variables)



# Models: composing layers

class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters
        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2c(x)
        x = self.bn2c(x, training=training)
        x += input_tensor
        return tf.nn.relu(x)


block = ResnetIdentityBlock(1, [1, 2, 3])
print(block(tf.zeros([1, 2, 3, 3])))
print([x.name for x in block.variables])




 my_seq = tf.keras.Sequential([tf.keras.layers.Conv2D(1, (1, 1)),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Conv2D(2, 1,
                                                      padding='same'),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Conv2D(3, (1, 1)),
                               tf.keras.layers.BatchNormalization()])
my_seq(tf.zeros([1, 2, 3, 3]))





# ----------------------------------------------------------------------------------------------------------------------
# Custom training: walkthrough
# ----------------------------------------------------------------------------------------------------------------------
# https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough
#
# 1.Import and parse the data sets.
# 2.Select the type of model.
# 3.Train the model.
# 4.Evaluate the model's effectiveness.
# 5.Use the trained model to make predictions.


# Setup program

from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))



# Import and parse the training dataset

# Download the dataset
train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)
print("Local copy of the dataset file: {}".format(train_dataset_fp))


!head -n5 {train_dataset_fp}

# import pandas as pd
# data = pd.read_csv('/home/ywb/.keras/datasets/iris_training.csv')

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]
label_name = column_names[-1]
print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))


class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']


# Create a tf.data.Dataset
batch_size = 32
train_dataset = tf.contrib.data.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

features, labels = next(iter(train_dataset))
features
labels


plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')
plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()


def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels


train_dataset = train_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))
print(features[:5])



# Select the type of model

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])


predictions = model(features)
predictions[:5]

tf.nn.softmax(predictions[:5])

print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))



# Train the model
def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
l = loss(model, features, labels)
print("Loss test: {}".format(l))

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
global_step = tf.train.get_or_create_global_step()


loss_value, grads = grad(model, features, labels)
print("Step: {}, Initial Loss: {}".format(global_step.numpy(),
                                          loss_value.numpy()))
optimizer.apply_gradients(zip(grads, model.variables), global_step)
print("Step: {},         Loss: {}".format(global_step.numpy(),
                                          loss(model, features, labels).numpy()))


# Training loop

## Note: Rerunning this cell uses the same model variables
# keep results for plotting
train_loss_results = []
train_accuracy_results = []
num_epochs = 201
for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()
    # Training loop - using batches of 32
    for x, y in train_dataset:
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step)
        # Track progress
        epoch_loss_avg(loss_value)  # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)
    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))


fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')
axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)
axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()



# Evaluate the model's effectiveness

test_url = "http://download.tensorflow.org/data/iris_test.csv"
test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)


test_dataset = tf.contrib.data.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)
test_dataset = test_dataset.map(pack_features_vector)



test_accuracy = tfe.metrics.Accuracy()
for (x, y) in test_dataset:
    logits = model(x)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)
print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


tf.stack([y,prediction],axis=1)




# Use the trained model to make predictions

predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5,],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]
])
predictions = model(predict_dataset)
for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[class_idx]
    name = class_names[class_idx]
    print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))





















































































































































































