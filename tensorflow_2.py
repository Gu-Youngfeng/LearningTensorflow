#!/usr/bin/python
# coding=utf-8

import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_lstm(train_set, test_set):