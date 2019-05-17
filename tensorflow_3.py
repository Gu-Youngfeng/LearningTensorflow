#!/usr/bin/python
# coding=utf-8

import tensorflow as tf
import csv_reader
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
hyper-parameter setting
"""
batch_size = 20
learning_rate = 0.001
learning_round = 1000
regular_lambda = 0.01

def build_cnn(train_set, test_set):
	"""
	build a simple cnn
	"""
	pass


if __name__ == "__main__":
	train_set, test_set = csv_reader.read_from_path_by_ratio("data/CM1.csv", 0.9)
	build_cnn(train_set, test_set)