import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


def mnist_dataset():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    # normalize
    x = x / 255.0
    x_test = x_test / 255.0

    return (x, y), (x_test, y_test)


class myModel:
    def __init__(self):
        self.W1 = tf.Variable(shape=[28 * 28, 100], dtype=tf.float32,
                              initial_value=tf.random.uniform(shape=[28 * 28, 100],
                                                              minval=-0.1, maxval=0.1))
        self.b1 = tf.Variable(shape=[100], dtype=tf.float32, initial_value=tf.zeros(100))  # f(x)
        self.W2 = tf.Variable(shape=[100, 10], dtype=tf.float32,  # wx+b
                              initial_value=tf.random.uniform(shape=[100, 10],
                                                              minval=-0.1, maxval=0.1))
        self.b2 = tf.Variable(shape=[10], dtype=tf.float32, initial_value=tf.zeros(10))
        self.trainable_variables = [self.W1, self.W2, self.b1, self.b2]

    def __call__(self, x):
        flat_x = tf.reshape(x, shape=[-1, 28 * 28])
        h1 = tf.tanh(tf.matmul(flat_x, self.W1) + self.b1)
        logits = tf.matmul(h1, self.W2) + self.b2
        return logits


model = myModel()

optimizer = optimizers.Adam()

@tf.function
def compute_loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))

@tf.function
def compute_accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

@tf.function
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(logits, y)

    # compute gradient
    trainable_vars = [model.W1, model.W2, model.b1, model.b2]
    grads = tape.gradient(loss, trainable_vars)
    for g, v in zip(grads, trainable_vars):
        v.assign_sub(0.01*g)

    accuracy = compute_accuracy(logits, y)

    # loss and accuracy is scalar tensor
    return loss, accuracy

@tf.function
def test(model, x, y):
    logits = model(x)
    loss = compute_loss(logits, y)
    accuracy = compute_accuracy(logits, y)
    return loss, accuracy

train_data, test_data = mnist_dataset()
for epoch in range(50):
    loss, accuracy = train_one_step(model, optimizer,
                                    tf.constant(train_data[0], dtype=tf.float32),
                                    tf.constant(train_data[1], dtype=tf.int64))
    print('epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())
loss, accuracy = test(model,
                      tf.constant(test_data[0], dtype=tf.float32),
                      tf.constant(test_data[1], dtype=tf.int64))

print('test loss', loss.numpy(), '; accuracy', accuracy.numpy())