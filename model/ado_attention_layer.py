# -*- coding:utf-8 -*-
"""
@contact: adonis_wu@outlook.com
@file: ado_embedding_layer.py
@time: 2018/11/12 15:46
"""
__author__ = '🍊 Adonis Wu 🍊'
import tensorflow as tf


class AttentionLayer(tf.layers.Layer):
    def __init__(self, hidden_size, num_heads, attention_dropout, train):
        if hidden_size % num_heads != 0:
            raise ValueError('hidden size must be evenly divisible by the number of heads.')

        super(AttentionLayer, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = hidden_size // num_heads
        self.attention_dropout = attention_dropout
        self.train = train

        # layers for linearly projection the query, key and value.
        self.Q_dense_layer = tf.layers.Dense(self.hidden_size, use_bias=False, name='Q')
        self.K_dense_layer = tf.layers.Dense(self.hidden_size, use_bias=False, name='K')
        self.V_dense_layer = tf.layers.Dense(self.hidden_size, use_bias=False, name='V')
        self.output_layer = tf.layers.Dense(self.hidden_size, use_bias=False, name='output_transform')


    def call(self, x, y, bias, cache=None, **kwargs):
        """
        :param inputs:
                x = inputs.get('x') a tensor with shape [batch_size, length_x, hidden_size]
                y = inputs.get('y') a tensor with shape [batch_size, length_y, hidden_size]
        :param kwargs:
        :return:
        """
        Q = self.Q_dense_layer(x)
        K = self.K_dense_layer(y)
        V = self.V_dense_layer(y)

        if cache is not None:
            K = tf.concat([cache['k'], K], axis=1)
            V = tf.concat([cache['v'], V], axis=1)

            cache['k'] = K
            cache['v'] = V
        # [batch_size, num_heads, length, hidden_size/num_heads]
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        #   scale Q to prevent the dot product between Q and K from growing too large.
        """
            a = tf.constant([[0, 2, 4, 6, 8, 9], [1, 3, 4, 5, 6, 7]], dtype=tf.float32)

            a *= 2 ** -0.5

            [[0.         1.4142135  2.828427   4.2426405  5.656854   6.3639607 ]
             [0.70710677 2.1213202  2.828427   3.535534   4.2426405  4.9497476 ]]
        """
        Q *= self.depth ** -0.5

        """
            Q               [batch_size, num_heads, length, hidden_size/num_heads]
            K               [batch_size, num_heads, length, hidden_size/num_heads] 
            K_transpose     [batch_size, num_heads, hidden_size/num_heads，length]
            logits          [batch_size, num_heads，length_x, length_y]
        """
        logits = tf.matmul(Q, K, transpose_b=True)

        logits = tf.add(logits, bias)

        """
            softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
        """
        weights = tf.nn.softmax(logits, name='attention_weights')

        if self.train:
            weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)

        """
            weights             [batch_size, num_heads，length_x, length_y]
            V                   [batch_size, num_heads，length_y, hidden_size/num_heads]
            attention_output    [batch_size, num_heads，length_x, hidden_size/num_heads]
        """
        attention_output = tf.matmul(weights, V)

        #   [batch_size, length_x, hidden_size]
        attention_output = self.combine_heads(attention_output)

        #   [batch_size, length_x, hidden_size]
        attention_output = self.output_layer(attention_output)

        return attention_output

    def split_heads(self, x):
        """
        :param x: a tensor with shape [batch_size, length, hidden_size]
        :return:  a tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
        """
        with tf.name_scope('split_heads'):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            x = tf.reshape(x, [batch_size, length, self.num_heads, self.depth])

            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """
        :param x:  a tensor [batch_size, num_heads, length, hidden_size/num_heads]
        :return:  a tensor [batch_size, length, hidden_size]
        """

        with tf.name_scope('combine_heads'):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(x, [0, 2, 1, 3])

            return tf.reshape(x, [batch_size, length, self.hidden_size])


class SelfAttention(AttentionLayer):
    """Multi head self-attention layer."""

    def call(self, x, bias, cache=None):
        return super(SelfAttention, self).call(x, x, bias, cache)
