#!/usr/bin/env python3
# conding=utf-8

import tensorflow as tf
import os
import numpy as np

def forward_propagation():
	"""
	Here we simulate the forward propagation of the following network,
	
	x1 		a11		
	     -> a12 ->	Y
	x2		a13

	"""
	w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
	w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))
	x = tf.constant([[0.7, 0.9]])

	a = tf.matmul(x, w1)
	y = tf.matmul(a, w2)
	with tf.Session() as sess:		
		# sess.run(w1.initializer)
		# sess.run(w2.initializer)
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		sess.run(y)
		print(y.eval())


def forward_propagation_placeholder():
	"""
	Error: InvalidArgumentError (see above for traceback): 
	You must feed a value for placeholder tensor 'input' with dtype float and shape [3,2]
	[Node: input = Placeholder[dtype=DT_FLOAT, shape=[3,2], _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]
	"""
	w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
	w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))
	# x = tf.constant([[0.7, 0.9]])
	x = tf.placeholder(tf.float32, shape=(3,2), name="input")

	a = tf.matmul(x, w1)
	y = tf.matmul(a, w2)
	with tf.Session() as sess:		
		# sess.run(w1.initializer)
		# sess.run(w2.initializer)
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]})
		print(y.eval())


def some_tf_functions():
	arr1 = tf.constant([[1.0,3.0,5.0], [4.0, 6.0, 8.0]], dtype=tf.float32, shape=(2,3))
	a = 2.718281828459
	with tf.Session() as sess:
		y_1 = tf.clip_by_value(arr1, 3.0, 7.0) # compress the vector arr1 to the range between 3.0 and 7.0.
		y_2 = tf.log(a)
		sess.run(y_1)
		sess.run(y_2)
		print(y_1.eval())
		print(y_2.eval())


if __name__ == "__main__":
	os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
	some_tf_functions()
