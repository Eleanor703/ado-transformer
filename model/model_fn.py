# -*- coding:utf-8 -*-
"""
@contact: adonis_wu@outlook.com
@file: model_fn.py
@time: 2018/11/13 15:29
"""
__author__ = '🍊 Adonis Wu 🍊'
import tensorflow as tf
from model.ado_transformer import Transformer
from utils import metrics
from model import model_utils


def model_fn(features, labels, mode: tf.estimator.ModeKeys, params: dict):
    """
    :param features:
                    encode_inputs = features['encode_feature_name']
    :param labels:
    :param mode:
    :param params:
    :return:
    """
    with tf.variable_scope('model'):
        inputs = features
        transformer = Transformer(params, mode == tf.estimator.ModeKeys.TRAIN)

        logits = transformer(inputs, labels)

        """
            when in prediction mode, the labels and decode_inputs  is None,
            the model output id the prediction
            it is a dict {"outputs": top_decoded_ids, "scores": top_scores}
        """

        if mode == tf.estimator.ModeKeys.PREDICT:
            estimator = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=logits,
                export_outputs={
                    'translate': tf.estimator.export.PredictOutput(logits)
                }
            )

            return estimator

        logits.set_shape(labels.shape.as_list() + logits.shape.as_list()[2:])

        xentropy, weights = metrics.padded_cross_entropy_loss(
            logits=logits,
            labels=labels,
            smoothing=params.get('label_smoothing'),
            vocab_size=params.get('vocab_size')
        )

        loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

        tf.identity(loss, 'cross_entropy')

        if mode == tf.estimator.ModeKeys.EVAL:
            estimator = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                predictions={'predictions': logits},
                eval_metric_ops=metrics.get_eval_metrics(logits, labels, params)
            )

            return estimator

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op, metrics_dict = model_utils.get_train_op_and_metrics(loss, params)

            metrics_dict['mini_batch_loss'] = loss

            model_utils.record_scalars(metrics_dict)

            estimator = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )

            return estimator
