#!/usr/bin/python
# coding=utf-8

"""
This python file provides the reader for sequential dataset
"""
import pandas as pd
import numpy as np
import os

def extract_sequence(txt_line):
	# is txt_line valid?
	if txt_line == "" or "," not in txt_line:
		print("[ERROR]: CANNOT EXTRACT FEATURES IN LINE:", txt_line)
		return None

	parts = txt_line.strip().split(',')
	sequence = [float(part) for part in parts if part != '' and len(part) != 0]
	return sequence	


def read_seq_from_txt(path):
	# is path valid?
	if os.path.exists(path)==False:
		print("[ERROR]: CANNOT FIND DATASET IN PATH:", path)
		return None

	file = open(path, 'r')
	# feature and label set
	features = []
	labels = []
	# flag
	flag = False
	sequences = []

	try:
		while True:
			txt_line = file.readline()
			if txt_line:
				if txt_line.startswith("/"):
					flag = True
					continue
				if flag:
					if txt_line.startswith("OutTrace"):
						labels.append([0.0])
						features.append(sequences)
						sequences = []
						flag = False
					elif txt_line.startswith("InTrace"):
						labels.append([1.0])
						features.append(sequences)
						sequences = []
						flag = False
					else:
						sequences.append(extract_sequence(txt_line))
			else:
				break
	finally:
		file.close()

	return [features, labels]


if __name__ == "__main__":
	features, labels = read_seq_from_txt("data/CM2.txt")
	print(features)
	print(labels)