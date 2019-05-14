#!/usr/bin/python
# coding=utf-8

"""
This python file provides the reader for csv dataset
"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler


class Sample(object):
	"""
	save each sample in the form of an object
	"""
	def __init__(self, features, label, predicted=-1):
		self.features = features
		self.label = label
		self.predicted = predicted


def normalize(samples, norm_type=0):
	# is samples empty?
	if len(samples) == 0:
		print("[ERROR]: CANNOT NORMALIZAE THE DATASET BECAUSE THE SAMPLES IS EMPTY.")
		return None

	tol_size = len(samples)
	fea_size = len(samples[0].features)

	train_matrix = []
	for i in range(tol_size):
		train_matrix.append(samples[i].features)

	train_matrix_normal = []
	if norm_type == 0:
		train_matrix_normal = MinMaxScaler().fit_transform(train_matrix)
	elif norm_type == 1:
		train_matrix_normal = StandardScaler().fit_transform(train_matrix)
	else:
		train_matrix_normal = MaxAbsScaler().fit_transform(train_matrix)

	for i in range(tol_size):
		samples[i].features = train_matrix_normal[i]

	return samples


def read_from(path, ratio):
	# is path valid?
	if os.path.exists(path)==False:
		print("[ERROR]: CANNOT FIND DATASET IN PATH:", path)
		return None
	# is ratio valid?
	if ratio >= 1 or ratio <= 0:
		print("[ERROR]: CANNOT SPLIT THE DATASET BY INVALID RATIO:", ratio)
		return None

	content = pd.read_csv(path)
	features = content.columns[:-1].tolist()
	labels = content.columns[-1]
	tol_size = len(content)

	print("# DATASET INFORMATION\n------------------------------------------")
	print("[PATH]:    ", path)
	print("[FEATURES]:",features)
	print("[LABELS]:  ", labels)
	print("[SIZE]:    ", tol_size)

	samples = []
	for i in range(tol_size):
		sample = Sample(content.iloc[i][:-1].tolist(), content.iloc[i][-1])
		samples.append(sample)

	# normalization
	samples = normalize(samples)
	# randomization
	np.random.shuffle(samples)

	train_size = int(tol_size*ratio)
	test_size = tol_size - train_size
	
	train_set = samples[:train_size]
	test_set = samples[train_size:tol_size]

	return [train_set, test_set]


if __name__ == "__main__":
	read_from("data/CM1.csv",0.9)