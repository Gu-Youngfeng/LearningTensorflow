#!/usr/bin/python
# coding=utf-8

import tensorflow as tf
import csv_reader
import matplotlib.pyplot as plt
from scipy.misc import imread, imshow, imresize
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
hyper-parameter setting
"""
batch_size = 20
learning_rate = 0.001
learning_round = 1000
regular_lambda = 0.01


def show_original_img(path):
	"""
	to show the image
	"""
	# read the 3-dimensional array from the image
	img = imread(path, flatten=False)
	# output the array
	print(img.shape)
	print(img)

	plt.imshow(img)
	plt.title("Peppa Pig "+str(img.shape))
	# remove the axis
	# plt.axis("off")
	plt.show()


def show_resied_img(path):
	"""
	to resize the image
	"""
	img_puppy = imread(path, flatten=False) 
	# resize the image
	img_puppy_new = imresize(img_puppy, size=(300, 300))
	print(img_puppy_new.shape)
	plt.imshow(img_puppy_new)
	plt.title("Peppa Pig (Resized) "+str(img_puppy_new.shape))
	plt.show()


def show_channels_img(path):
	"""
	to split the R/G/B channels of the image 
	"""
	img_puppy = imread(path, flatten=False)
	# channel RED
	puppy_R = np.zeros(img_puppy.shape)
	puppy_R[:,:,0] = img_puppy[:,:,0]
	# channel GREEN
	puppy_G = np.zeros(img_puppy.shape)
	puppy_G[:,:,1] = img_puppy[:,:,1]
	# channel BLUE
	puppy_B = np.zeros(img_puppy.shape)
	puppy_B[:,:,2] = img_puppy[:,:,2]

	# RED
	plt.subplot(1,3,1)
	plt.imshow(puppy_R)
	plt.axis("off")
	plt.title("(RED)")
	# GREEN
	plt.subplot(1,3,2)
	plt.imshow(puppy_G)
	plt.axis("off")
	plt.title("(GREEN)")
	# BLUE
	plt.subplot(1,3,3)
	plt.imshow(puppy_B)
	plt.axis("off")
	plt.title("(BLUE)")
	plt.show()


def build_cnn(train_set, test_set):
	"""
	build a simple cnn
	"""
	pass


def try_cnn():
	img_arr = plt.imread("data/puppy.jpg")
	print(img_arr.shape)


	img_input = tf.placeholder(tf.float32, shape=(None, 200, 200, 3), name="pic_features")
	filter_weight = tf.Variable(tf.random_normal(shape=(5,5,3,16),stddev=0.1))
	biases = tf.Variable(tf.random_normal(shape=(16,),stddev=0.1))

	conv = tf.nn.conv2d(img_input, filter_weight, strides=[1,1,1,1], padding="VALID")
	bias = tf.nn.bias_add(conv, biases)
	actived_conv = tf.nn.relu(bias)

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		print(sess.run(actived_conv, feed_dict={img_input: [img_arr]}))


if __name__ == "__main__":
	"""
	All the pictures in color should be recorded as the 3-dimensional arraies
	img_puppy.shape = (pixel_high, piexl_length, 3)
	"""
	# show_original_img("data/puppy.jpg")
	# show_resied_img("data/puppy.jpg")
	# show_channels_img("data/puppy.jpg")

	"""
	"""
	try_cnn()


	# train_set, test_set = csv_reader.read_from_path_by_ratio("data/CM1.csv", 0.9)
	# build_cnn(train_set, test_set)

	# import numpy as np
	# data = np.array([[[1,2,1],[2,1,0]],[[0,-1,2],[-1,-2,1]]])
	# print(data)
	# print("------------")
	# print("Layer R:\n", data[:,:,0])
	# print("Layer G:\n", data[:,:,1])
	# print("Layer B:\n", data[:,:,2])