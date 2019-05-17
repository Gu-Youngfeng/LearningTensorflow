#!/usr/bin/python
# coding=utf-8

import tensorflow as tf
import seq_reader
import os 
import numpy as np
from keras.preprocessing.sequence import pad_sequences
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
hyper-peremeter setting
"""
lstm_hidden_size = 128
batch_size = 4
sequence_size = 10
feature_size = 45
learning_rate = 0.01
learning_round = 20

def cal_length(seq):
	used = tf.sign(tf.reduce_max(tf.abs(seq), 2))
	length = tf.reduce_sum(used, 1)
	length = tf.cast(length, tf.int32)
	return length


def build_simple_lstm(features_train, labels_train, features_test, labels_test):
	# train set input transformation
	features_train = np.array(features_train)
	features_train = pad_sequences(features_train, padding='post', maxlen=sequence_size) # padding with 0
	labels_train = np.array(labels_train)
	# train size
	train_size = len(labels_train)

	# test set input transformation
	features_test = np.array(features_test)
	features_test = pad_sequences(features_test, padding='post', maxlen=sequence_size) # padding with 0
	labels_test = np.array(labels_test)
	# test size
	test_size = len(labels_test)

	# x has the shape of (4, 10, 45)
	x = tf.placeholder(tf.float32, shape=(None, sequence_size, feature_size), name="features")
	# y has the shape of (None, 1)
	y = tf.placeholder(tf.float32, shape=(None,1), name="labels")
	# batch size of sequential data
	# place_batch_size = tf.placeholder(tf.int32, [], name='batch_size')

	# lstm cell
	# here tf.nn.rnn_cell.BasicLSTMCell() and tf.nn.rnn_cell.LSTMCell() are both ok, 
	# refer to https://tensorflow.google.cn/api_docs/python/tf/nn/rnn_cell/BasicLSTMCell
	lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_hidden_size)
	# initialize to zero
	# init_state = lstm_cell.zero_state(batch_size=place_batch_size, dtype=tf.float32)
	# dynamic rnn
	# outputs, state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=x, initial_state=init_state, sequence_length=cal_length(x), dtype=tf.float32)
	outputs, state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=x, sequence_length=cal_length(x), dtype=tf.float32)

	# select the last lstm cell's output as our final output
	h=outputs[:,-1,:]
	# h = tf.reshape(outputs, [-1, lstm_hidden_size])

	# initialize the parameter with random uniform distribution 
	weights = tf.Variable(tf.random_uniform(shape=(128,1), minval=0, maxval=1), dtype=tf.float32)
	biase = tf.Variable(tf.random_uniform(shape=(1,1), minval=0, maxval=1), dtype=tf.float32)
	# initialize the parameter with random normal distribution
	# weights = tf.Variable(tf.random_normal(shape=(128,1), stddev=1e-3), dtype=tf.float32)
	# biase = tf.Variable(tf.random_normal(shape=(1,1), stddev=1e-3), dtype=tf.float32)

	y_predicted = tf.nn.relu(tf.matmul(h, weights) + biase)

	# loss funtion
	cross_entropy = tf.reduce_mean(tf.square(y - y_predicted))
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

	# define the accuaracy and other measurements
	y_predicted = tf.cast(tf.not_equal(y_predicted, 0.0), tf.float32)

	# define the tp, tn, fp and fn
	tp = tf.reduce_sum(tf.multiply(y, y_predicted))
	tn = tf.reduce_sum(tf.cast(tf.equal(y,y_predicted), tf.float32)) - tp
	fp = tf.reduce_sum(tf.cast(tf.greater(y_predicted, y), tf.float32))
	fn = tf.reduce_sum(tf.cast(tf.greater(y, y_predicted), tf.float32))
	# define the precision, recall and fmeasure in class 1.0
	precision_1 = tp/(tp + fp)
	recall_1 = tp/(tp + fn)
	fmeasure_1 = precision_1*recall_1*2/(precision_1 + recall_1)
	# define the precision, recall and fmeasure in class 0.0
	precision_0 = tn/(tn + fn)
	recall_0 = tn/(tn + fp)
	fmeasure_0 = precision_0*recall_0*2/(precision_0 + recall_0)
	# define the accuracy
	accuracy = (tp+tn)/(tp+tn+fn+fp)

	init = tf.global_variables_initializer()
	entropy_train = []
	entropy_test = []
	with tf.Session() as sess:
		sess.run(init)

		for i in range(learning_round):
			start_index = (batch_size*i)%train_size
			end_index = min(start_index+batch_size, train_size)
			# train the train_step
			# sess.run(train_step, feed_dict={x:features_train[start_index:end_index], y:labels_train[start_index:end_index], place_batch_size:batch_size})
			sess.run(train_step, feed_dict={x:features_train[start_index:end_index], y:labels_train[start_index:end_index]})
			# calculate the entropy
			# train_cross_entropy = sess.run(cross_entropy, feed_dict={x:features_train, y:labels_train, place_batch_size:train_size}) 
			train_cross_entropy = sess.run(cross_entropy, feed_dict={x:features_train, y:labels_train}) 
			entropy_train.append(train_cross_entropy)
			test_cross_entropy = sess.run(cross_entropy, feed_dict={x:features_test, y:labels_test})
			entropy_test.append(test_cross_entropy)
			# if i%50 == 0:
			print("ROUND:", i, "LOSS:", train_cross_entropy)

		# prdict the test set
		precision_1, recall_1, fmeasure_1, precision_0, recall_0, fmeasure_0, accuracy = sess.run([precision_1, recall_1, fmeasure_1, precision_0, recall_0, fmeasure_0, accuracy], 
																									feed_dict={x:features_test, y:labels_test})
		print("[PREDICTION]:", precision_1, recall_1, fmeasure_1, precision_0, recall_0, fmeasure_0, accuracy)
		# print the predicted labels and real labels
		print(sess.run(y_predicted, feed_dict={x:features_test, y:labels_test}))
		print("-----")
		print(labels_test)
	
	# draw the trade of cross_entropy with learining round
	import matplotlib.pyplot as plt
	x = range(len(entropy_train))
	plt.plot(x, entropy_train)
	plt.plot(x, entropy_test)
	plt.xlabel("Learning rounds")
	plt.ylabel("Cross entropy")
	plt.legend(["entropy on the training set", "entropy on the testing set"])
	plt.show()


if __name__ == "__main__":
	# get the test set and the train set
	features_train, labels_train = seq_reader.read_seq_from_txt("data/CM2.txt")
	features_test, labels_test = seq_reader.read_seq_from_txt("data/CM2.txt")
	# build model on train set and predict on the test set
	build_simple_lstm(features_train, labels_train, features_test, labels_test)


