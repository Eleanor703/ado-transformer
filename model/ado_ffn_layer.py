# -*- coding:utf-8 -*-
"""
@contact: adonis_wu@outlook.com
@file: ado_ffn_layer.py
@time: 2018/11/12 17:55
"""
__author__ = '🍊 Adonis Wu 🍊'
import tensorflow as tf


class FFNLayer(tf.layers.Layer):
    def __init__(self, hidden_size, filter_size, relu_dropout, train, allow_pad):
        super(FFNLayer, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.train = train
        self.allow_pad = allow_pad
        self.filter_layer = tf.layers.Dense(
            units=self.filter_size,
            use_bias=True,
            name='filter_layer'
        )

        self.output_layer = tf.layers.Dense(
            units=self.hidden_size,
            use_bias=True,
            name='output_layer'
        )

    def call(self, inputs, padding=None):
        """
        :param inputs: a tensor with shape [batch_size, length, hidden_size]
        :param padding: [batch_size, length]
        :return: [batch_size, length, hidden_size]
        """

        padding = padding if self.allow_pad else None

        batch_size = tf.shape(inputs)[0]

        length = tf.shape(inputs)[1]

        if padding is not None:
            with tf.name_scope('remove_padding'):
                pad_mask = tf.reshape(padding, [-1])
                """
                    a = tf.constant([1, 2, 3, 0, 5])
                    b = tf.where(a < 2)
                    c = tf.to_int32(b)
                    
                    b = tf.Tensor([[0][3]], shape=(2, 1), dtype=int64)
                    c = tf.Tensor([[0][3]], shape=(2, 1), dtype=int32)
                    
                    non_pad_ids [<= batch_size * length, 1]    
                        1 dimension stands for index 
                        lower stands how many size which less than 1e-9, this size must <= batch_size * length
                        
                """
                non_pad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))
                inputs = tf.reshape(inputs, [-1, self.hidden_size])
                """
                    indices = [[1], [0]]
                    params = [['a', 'b'], ['c', 'd']]
                    output = [['c', 'd'], ['a', 'b']]
                
                    inputs          [batch_size * length, hidden_size]
                    non_pad_ids     [<= batch_size * length, 1]
                    inputs          [<= batch_size * length, hidden_size]
                """
                inputs = tf.gather_nd(params=inputs, indices=non_pad_ids)

                inputs.set_shape([None, self.hidden_size])
                #   [1, <= batch_size * length, hidden_size]
                inputs = tf.expand_dims(inputs, axis=0)

        outputs = self.filter_layer(inputs)

        if self.train:
            outputs = tf.nn.dropout(outputs, 1.0 - self.relu_dropout)

        outputs = self.output_layer(outputs)

        if padding is not None:
            with tf.name_scope('re_add_padding'):
                outputs = tf.squeeze(outputs, axis=0)
                outputs = tf.scatter_nd(
                    indices=non_pad_ids,
                    updates=outputs,
                    shape=[batch_size * length, self.hidden_size]
                )

                outputs = tf.reshape(outputs, [batch_size, length, self.hidden_size])

        return outputs
