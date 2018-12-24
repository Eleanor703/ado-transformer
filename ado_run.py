# -*- coding:utf-8 -*-
"""
@contact: adonis_wu@outlook.com
@file: ado_train.py
@time: 2018/11/1 16:58
"""
__author__ = '🍊 Adonis Wu 🍊'
import tensorflow as tf

import utils.ado_utils as util
from model.ado_input import AdoInput
from model.model_fn import model_fn


def train():
    params = util.load_config('config.yml', 'transformer_01')

    #   enrich vocabs
    util.enrich_params(params)

    log_level = 'tf.logging.{}'.format(str(params.get('log_level')).upper())
    tf.logging.set_verbosity(eval(log_level))

    #   config check_steps and max to keep
    config = util.load_sess_config(params)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=params.get('model_dir'),
        config=config,
        params=params,
    )

    input = AdoInput(params)
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input.input_fn(mode=tf.estimator.ModeKeys.TRAIN, vocabs=params.get('vocabs')),
        max_steps=params.get('max_steps', None),
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input.input_fn(mode=tf.estimator.ModeKeys.EVAL, vocabs=params.get('vocabs')),
        throttle_secs=params.get('throttle_secs'),
    )

    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec,
        eval_spec=eval_spec
    )

    predictions = estimator.predict(
        input_fn=lambda: input.input_fn(mode=tf.estimator.ModeKeys.PREDICT, vocabs=params.get('vocabs')),
    )

    for index, prediction in enumerate(predictions):
        label_c = prediction['classes'][0].decode()
        logits_c = prediction['logits'].tolist()
        print('logits {} -> label {}'.format(logits_c, label_c))


if __name__ == '__main__':
    train()
