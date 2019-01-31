#!/usr/bin/env python3
# coding=utf-8

import tensorflow as tf
import os

def session_start_1():
	sess = tf.Session()
	# some operations
	sess.run()
	sess.close()

def session_start_2():
	"""
	The normal start of a session
	"""
	config = tf.ConfigProto(allow_soft_place_placement=True, # allow to change scheme from GPU to CPU
							log_device_placement=True) # record the log when runing the session
	sess = tf.Session(config = config)
	# some operations
	sess.run()
	sess.close()

def session_start_3():
	"""
	The 'with' statement will release the resource when the exception happens during the run() process.
	"""
	with tf.Session() as sess:
		# some operations
		sess.run()

def session_start_4():
	with tf.Session() as sess:
		a = tf.constant([1, 2, 3], name="a")
		b = tf.constant([1, 2, 3], name="b")
		result = a + b
		print(result) # tensor vector
		print(result.eval()) # value


if __name__ == "__main__":
	#os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息  
	#os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error   
	os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error
	session_start_4()
