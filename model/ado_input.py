# -*- coding:utf-8 -*-
"""
@contact: adonis_wu@outlook.com
@file: ado_input.py
@time: 2018/11/7 16:25
"""
__author__ = '🍊 Adonis Wu 🍊'
import json
import os

import tensorflow as tf

import utils.ado_data_utils as utils


class AdoInput():

    def __init__(self, params: dict):
        self.data_dir = params.get('data_dir')
        self.encode_max_length = params.get('max_encode_length')
        self.decode_max_length = params.get('max_decode_length')
        self.batch_size = params.get('batch_size')

    def get_data_dir(self, mode: tf.estimator.ModeKeys):

        if mode == tf.estimator.ModeKeys.TRAIN:
            self.file_pattern = '*.train.*'
            sub = 'train'
        elif mode == tf.estimator.ModeKeys.EVAL:
            self.file_pattern = '*.eval.*'
            sub = 'eval'
        else:
            self.file_pattern = '*.test.*'
            sub = 'test'

        return os.path.join(self.data_dir, sub, self.file_pattern)

    def input_fn(self, mode: tf.estimator.ModeKeys, vocabs: list):

        files = self.get_data_dir(mode)

        file_names = tf.gfile.Glob(files)

        ds = tf.data.TextLineDataset(file_names)

        def parse(raw):
            raw = raw.decode('utf-8')
            dicts = json.loads(raw)

            content = dicts['content']

            if mode == tf.estimator.ModeKeys.PREDICT:
                encode_ids = utils.segment_predict(content, vocabs, self.encode_max_length)
                return encode_ids
            else:
                title = dicts['title']
                encode_ids, decode_ids = utils.segment_train(content, title, vocabs, self.encode_max_length,
                                                             self.decode_max_length)
                return encode_ids, decode_ids

        if mode == tf.estimator.ModeKeys.PREDICT:
            ds = ds \
                .repeat(1) \
                .shuffle(buffer_size=20 * 1000) \
                .map(lambda line: tf.py_func(parse, inp=[line], Tout=[tf.int64])) \
                .map(lambda encode_idx: tf.reshape(encode_idx, [self.encode_max_length])) \
                .batch(self.batch_size) \
                .prefetch(buffer_size=20 * 1000)
        else:
            ds = ds \
                .repeat(None if mode == tf.estimator.ModeKeys.TRAIN else 1) \
                .shuffle(buffer_size=20 * 1000) \
                .map(lambda line: tf.py_func(parse, inp=[line], Tout=[tf.int64, tf.int64])) \
                .map(lambda encode_idx, decode_idx: (
                tf.reshape(encode_idx, [self.encode_max_length]),
                tf.reshape(decode_idx, [self.decode_max_length]))) \
                .batch(self.batch_size) \
                .prefetch(buffer_size=20 * 1000)
        return ds
