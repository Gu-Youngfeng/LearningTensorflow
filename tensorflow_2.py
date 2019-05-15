#!/usr/bin/python
# coding=utf-8

import tensorflow as tf
import seq_reader
import os 
# from keras.preprocessing.sequence import pad_sequences
# features_train_padded = pad_sequences(features_train, padding='post', maxlen=sequence_size)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
hyper-peremeter setting
"""
lstm_hidden_size = 128
batch_size = 4
sequence_size = 10
feature_size = 45
learning_rate = 0.01
learning_round = 10

def build_simple_lstm(features_train, labels_train):
	# x has the shape of (4, 10, 45)
	x = tf.placeholder(tf.float32, shape=(batch_size, sequence_size, feature_size), name="features")
	# y has the shape of (None, 1)
	y = tf.placeholder(tf.float32, shape=(None,1), name="labels")
	# fixed length of sequential data
	seq_size = tf.placeholder(tf.int32)

	# lstm cell
	lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_hidden_size)
	# initialize to zero
	init_state = lstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
	# dynamic rnn
	outputs, state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=x, sequence_length=seq_size, dtype=tf.float32)
	print(tf.shape(outputs))
	# output shape
	h=outputs[:,-1,:]

	# loss funtion
	cross_entropy = tf.losses.mean_squared_error(labels=y, predictions=h)
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		for i in range(learning_round):
			start_index = (batch_size*i)%train_size
			end_index = min(start_index+batch_size, train_size)

			sess.run(train_step, feed_dict={x:features_train[start_index:end_index], y:labels_train[start_index:end_index], seq_size:sequence_size})

			train_cross_entropy = sess.run(cross_entropy, feed_dict={x:features_train, y:features_train, seq_size:sequence_size})
			entropy_train.append(train_cross_entropy)
			# test_cross_entropy = sess.run(cross_entropy, feed_dict={x:test_features_matrix, y:test_labels_matrix})
			# entropy_test.append(test_cross_entropy)
			if i%50 == 0:
				print("ROUND:", i, "LOSS:", train_cross_entropy)


if __name__ == "__main__":
	features_train, labels_train = seq_reader.read_seq_from_txt("data/CM2.txt")
	build_simple_lstm(features_train, labels_train)


