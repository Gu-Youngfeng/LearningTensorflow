#!/usr/bin/python
#coding:utf-8

"""
This python file provides:
	1. the usages of libraries np.random and tf.random
	2. the usages of tensorbord graph
	3. the example of simple network
"""
import numpy as np
import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def try_numpy():
	# 2X5 matrix with elems in [0,1)
	matrix_0 = np.random.rand(2,5) 
	print(matrix_0, "\n")
	"""
	matrix_0 = [[0.09007902 0.49225362 0.06559182 0.70318778 0.02683153]
				[0.5162675  0.71188176 0.98354108 0.41362371 0.35791622]]
	"""

	# 2X5 matrix with elems in [2,10]
	matrix_1 = np.random.randint(low=0, high=10, size=(2, 5)) 
	print(matrix_1, "\n")
	"""
	matrix_1 = [[2 0 3 3 6]
				[6 8 0 8 7]]
	"""

	# 2X5 matrix with normal distribution N(0,1)
	matrix_2 = np.random.randn(2,5) 
	print(matrix_2, "\n")
	"""
	matrix_2 = [[-0.15123423 -0.42248768 -1.07979524 -1.46344092 -0.17976437]
				[ 0.48459167 -0.97487083  0.94116145  0.42251838  0.26447006]] 
	"""

	# 2X5 matrix with binomial distribution
	matrix_3 = np.random.binomial(10, 0.5, size=(2, 5)) 
	print(matrix_3, "\n")
	"""
	matrix_3 = [[8 3 5 4 4]
				[4 3 5 7 4]]
	"""


def try_tensorflow():
	# uniform distribution
	matrix_0 = tf.Variable(tf.random_uniform(shape=(2, 5), minval=0, maxval=5, dtype=tf.float32))
	# normal distribution
	matrix_1 = tf.Variable(tf.random_normal(shape=(2, 5), mean=0, stddev=1, dtype=tf.float32))
	# all zeros
	matrix_2 = tf.Variable(tf.zeros(shape=(2, 5), dtype=tf.float32))
	# variables initialization
	init = tf.global_variables_initializer()
	# open a session
	with tf.Session() as sess:
		sess.run(init)
		print(sess.run(matrix_0), "\n")
		print(sess.run(matrix_1), "\n")
		print(sess.run(matrix_2), "\n")


def try_graph():
	with tf.name_scope("add"):
		a = tf.Variable(tf.random_normal(shape=(2,5), mean=0, stddev=1), name="input_1")
		b = tf.Variable(tf.random_uniform(shape=(2,5), minval=0, maxval=1), name="input_2")
		result = a + b
	# variables initialization
	init = tf.global_variables_initializer()
	# open a session
	with tf.Session() as sess:
		sess.run(init)
		print(sess.run(result))
		"""
		FileWriter will record the graph within a log-file in the vis_log/ folder.
		We only need to open them in the tensorbord in your browser
		Step-1: run "tensorboard --logdir=path/to/your/log-file" in terminate
		Step-2: visit "localhost:6006" in browser
		"""
		writer = tf.summary.FileWriter("vis_log/", sess.graph)
		writer.close()


def try_simple_network():
	x = tf.constant([[0.7, 0.9]], dtype=tf.float32) # this is a 1X2 matrix

	weight_0 = tf.Variable(tf.random_normal([2,3], mean=0, stddev=1), dtype=tf.float32)
	weight_1 = tf.Variable(tf.random_normal([3,1], mean=0, stddev=1), dtype=tf.float32)

	layer_0 = tf.matmul(x, weight_0)
	y = tf.matmul(layer_0, weight_1)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		print(sess.run(y))
		writer = tf.summary.FileWriter("vis_log/", sess.graph)
		writer.close()


if __name__ == "__main__":
	# test numpy
	# try_numpy()
	# test tensorflow
	# try_tensorflow()
	# test graph
	# try_graph()
	# test simple network
	try_simple_network()
