# work for two hidden layers
# picasso inverse kinematics type1

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt#输出结果可视化的模块

data=pd.read_csv("train_data1.txt",names=['y1','y2','y3','x1','x2','x3','z1','z2','z3'])
data=np.matrix(data)
y_data=data[:,:3]
x_data=data[:,3:6]

X = tf.placeholder(tf.float32, shape=[None, 3], name="X")
Y = tf.placeholder(tf.float32, shape=[None, 3], name="Y")

W1 = tf.Variable(tf.random_normal([3, 5]), name='weight1')
b1 = tf.Variable(tf.random_normal([5]), name='bias1')
layer1 = tf.tanh(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([5, 3]), name='weight2')
b2 = tf.Variable(tf.random_normal([3]), name='bias2')
hypothesis = tf.matmul(layer1, W2) + b2

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
   # Initialize TensorFlow variables
   sess.run(tf.global_variables_initializer())
   
   for step in range(10001):
       sess.run(train, feed_dict={X: x_data, Y: y_data})

       if step % 100 == 0:
           print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run([W1, W2]))

   # Accuracy report
   h, c, a = sess.run([hypothesis, predicted, accuracy],
                      feed_dict={X: x_data, Y: y_data})
   print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
   print(x_data)
   print(y_data)

