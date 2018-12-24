# -*- coding:utf-8 -*-
"""
@contact: adonis_wu@outlook.com
@file: ado_emb_layer.py
@time: 2018/11/12 16:52
"""
__author__ = '🍊 Adonis Wu 🍊'
import tensorflow as tf


class EmbLayer(tf.layers.Layer):

    def __init__(self, vocab_size, hidden_size):
        super(EmbLayer, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def build(self, _):
        with tf.variable_scope('embedding_and_softmax', reuse=tf.AUTO_REUSE):
            self.shared_weights = tf.get_variable(
                name='weights',
                shape=[self.vocab_size, self.hidden_size],
                initializer=tf.random_normal_initializer(mean=0., stddev=self.hidden_size ** -0.5)
            )

        self.built = True

    def call(self, inputs, **kwargs):
        """
        :param inputs: a tensor with shape [batch_size, length]
        :param kwargs:
        :return:a tensor with shape  [batch_size, length, hidden_size]
        """
        with tf.name_scope('embedding'):
            """
                a = tf.constant([1,2,3,0,5])
                b = tf.not_equal(a, 0)
                b = [ True  True  True  False  True]
                c = tf.to_float(b)
                c = [1. 1. 1. 0. 1.]
            """
            mask = tf.to_float(tf.not_equal(inputs, 0))

            embeddings = tf.gather(self.shared_weights, inputs)
            embeddings *= tf.expand_dims(mask, -1)

            #   scale embedding by the sqrt of the hidden size
            embeddings *= self.hidden_size ** 0.5

            return embeddings

    def linear(self, inputs):
        """
        :param inputs:  a tensor with shape [batch_size, length, hidden_size]
        :return: float32 tensor with shape [batch_size, length, vocab_size]
        """

        with tf.name_scope('pre_softmax_linear'):
            batch_size = tf.shape(inputs)[0]
            length = tf.shape(inputs)[1]

            inputs = tf.reshape(inputs, [-1, self.hidden_size])
            """
                inputs              [batch_size, length, hidden_size]
                shared_weights      [vocab_size, hidden_size]
                transpose           [hidden_size, vocab_size]
                logits              [batch_size, length, vocab_size]
            """
            logits = tf.matmul(inputs, self.shared_weights, transpose_b=True)

            return tf.reshape(logits, [batch_size, length, self.vocab_size])
