import tensorflow as tf
import numpy as np
def softmax(x):
    ##########
    '''实现softmax函数，只要求对最后一维归一化，
    不允许用tf自带的softmax函数'''
    ##########
    x_exp = tf.math.exp(x)
    x_sum = tf.reduce_sum(x_exp, axis=1, keepdims=True)
    prob_x = x_exp / x_sum
    return prob_x

test_data = np.random.normal(size=[10, 5])
print((softmax(test_data).numpy() - tf.nn.softmax(test_data, axis=-1).numpy())**2 <0.0001)

print('-----我是分割线-----')

def sigmoid(x):
    ##########
    '''实现sigmoid函数， 不允许用tf自带的sigmoid函数'''
    ##########
    prob_x = 1 / (1 + (1 / tf.math.exp(x)))
    return prob_x

test_data = np.random.normal(size=[10, 5])
print((sigmoid(test_data).numpy() - tf.nn.sigmoid(test_data).numpy())**2 < 0.0001)

print('-----我是分割线-----')

def softmax_ce(x, label):
    ##########
    '''实现 softmax 交叉熵loss函数， 不允许用tf自带的softmax_cross_entropy函数'''
    ##########
    losses = -tf.reduce_sum(label * tf.math.log(x), axis=1)
    loss = tf.reduce_mean(losses)
    return loss

test_data = np.random.normal(size=[10, 5])
prob = tf.nn.softmax(test_data)
label = np.zeros_like(test_data)
label[np.arange(10), np.random.randint(0, 5, size=10)]=1.
print(((tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(label, test_data))
  - softmax_ce(prob, label))**2 < 0.0001).numpy())

print('-----我是分割线-----')

def sigmoid_ce(x, label):
    ##########
    '''实现 softmax 交叉熵loss函数， 不允许用tf自带的softmax_cross_entropy函数'''
    ##########
    losses = -tf.reduce_sum(label*tf.math.log(x)+(1.-label)*tf.math.log(1-x))/len(x)
    loss = tf.reduce_mean(losses)
    return loss

test_data = np.random.normal(size=[10])
prob = tf.nn.sigmoid(test_data)
label = np.random.randint(0, 2, 10).astype(test_data.dtype)
print(((tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(label, test_data))- sigmoid_ce(prob, label))**2 < 0.0001).numpy()
)
