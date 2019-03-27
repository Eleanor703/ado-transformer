# -*- coding:utf-8 -*-
"""
@contact: adonis_wu@outlook.com
@file: ado_test.py
@time: 2018/11/12 16:11
"""
__author__ = '🍊 Adonis Wu 🍊'
import tensorflow as tf

tf.enable_eager_execution()
a = tf.ones(shape=[3,4])
b = tf.ones(shape=[1,4])

c = tf.add(a,b)

print(c)
