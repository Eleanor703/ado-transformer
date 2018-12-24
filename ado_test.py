# -*- coding:utf-8 -*-
"""
@contact: adonis_wu@outlook.com
@file: ado_test.py
@time: 2018/11/12 16:11
"""
__author__ = '🍊 Adonis Wu 🍊'
import tensorflow as tf
tf.enable_eager_execution()
A = tf.ones(shape=[2,3,4])
print(A)
b = tf.reshape(A, [2,3,2,2])
c = tf.transpose(b, [0,2,1,3])
print(c)
