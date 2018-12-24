# -*- coding:utf-8 -*-
"""
@contact: adonis_wu@outlook.com
@file: ado_transformer.py
@time: 2018/11/13 09:18
"""
__author__ = '🍊 Adonis Wu 🍊'

import math

import tensorflow as tf

_NEG_INF = -1e9


def get_position_encoding(
        length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.

    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.

    Args:
      length: Sequence length.
      hidden_size: Size of the
      min_timescale: Minimum scale that will be applied at each position
      max_timescale: Maximum scale that will be applied at each position

    Returns:
      Tensor with shape [length, hidden_size]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = hidden_size // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) / (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    """
        position            [length, 1]
        inv_timescales      [1, hidden_size // 2]
        scaled_time         [length, hidden_size // 2]
        signal              [length, hidden_size]
    """
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)

    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


def get_decoder_self_attention_bias(length):
    """Calculate bias for decoder that maintains model's auto regressive property.

    Creates a tensor that masks out locations that correspond to illegal
    connections, so prediction at position i cannot draw information from future
    positions.

    Args:
      length: int length of sequences in batch.

    Returns:
      float tensor of shape [1, 1, length, length]
    """
    with tf.name_scope("decoder_self_attention_bias"):
        """
            valid_locs
                [[1. 0. 0. 0. 0.]
                 [1. 1. 0. 0. 0.]
                 [1. 1. 1. 0. 0.]
                 [1. 1. 1. 1. 0.]
                 [1. 1. 1. 1. 1.]]
            valid_locs  [1,1,length,length]
        """
        valid_locs = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        decoder_bias = _NEG_INF * (1.0 - valid_locs)
    return decoder_bias


def get_padding(x, padding_value=0):
    """Return float tensor representing the padding values in x.

    Args:
      x: int tensor with any shape
      padding_value: int value that

    Returns:
      float tensor with same shape as x containing values 0 or 1.
        0 -> non-padding, 1 -> padding
    """
    with tf.name_scope("padding"):
        return tf.to_float(tf.equal(x, padding_value))


def get_padding_bias(x):
    """Calculate bias tensor from padding values in tensor.

    Bias tensor that is added to the pre-softmax multi-headed attention logits,
    which has shape [batch_size, num_heads, length, length]. The tensor is zero at
    non-padding locations, and -1e9 (negative infinity) at padding locations.

    Args:
      x: int tensor with shape [batch_size, length]

    Returns:
      Attention bias tensor of shape [batch_size, 1, 1, length].
    """
    with tf.name_scope("attention_bias"):
        padding = get_padding(x)
        attention_bias = padding * _NEG_INF
        attention_bias = tf.expand_dims(
            tf.expand_dims(attention_bias, axis=1), axis=1)
    return attention_bias


def get_train_op_and_metrics(loss, params):
    """Generate training op and metrics to save in TensorBoard."""
    with tf.variable_scope("get_train_op"):
        if params.get('use_dynamic_lr'):
            tf.logging.info('use_dynamic_lr')
            learning_rate = get_learning_rate(
                learning_rate=params["learning_rate"],
                hidden_size=params["hidden_size"],
                learning_rate_warmup_steps=params["learning_rate_warmup_steps"])
        else:
            tf.logging.info('use static learning rate')
            learning_rate = params["static_learning_rate"]
            
        tf.identity(learning_rate, name='learning_rate')

        # Create optimizer. Use LazyAdamOptimizer from TF contrib, which is faster
        # than the TF core Adam optimizer.
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=params["optimizer_adam_beta1"],
            beta2=params["optimizer_adam_beta2"],
            epsilon=eval(params["optimizer_adam_epsilon"])
        )
        # optimizer = tf.contrib.opt.LazyAdamOptimizer(
        #     learning_rate,
        #     beta1=params["optimizer_adam_beta1"],
        #     beta2=params["optimizer_adam_beta2"],
        #     epsilon=params["optimizer_adam_epsilon"])

        # Calculate and apply gradients using LazyAdamOptimizer.
        global_step = tf.train.get_global_step()
        variables = tf.trainable_variables()
        gradients = optimizer.compute_gradients(loss, variables, colocate_gradients_with_ops=True)
        minimize_op = optimizer.apply_gradients(
            gradients, global_step=global_step, name="train")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)

        train_metrics = {"learning_rate": learning_rate}

        # gradient norm is not included as a summary when running on TPU, as
        # it can cause instability between the TPU and the host controller.
        gradient_norm = tf.global_norm(list(zip(*gradients))[0])
        train_metrics["global_norm/gradient_norm"] = gradient_norm

        return train_op, train_metrics


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):
    """Calculate learning rate with linear warmup and rsqrt decay."""
    with tf.name_scope("learning_rate"):
        warmup_steps = tf.to_float(learning_rate_warmup_steps)
        step = tf.to_float(tf.train.get_or_create_global_step())

        learning_rate *= (hidden_size ** -0.5)
        # Apply linear warmup
        learning_rate *= tf.minimum(1.0, step / warmup_steps)
        # Apply rsqrt decay
        learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))

        # Create a named tensor that will be logged using the logging hook.
        # The full name includes variable and names scope. In this case, the name
        # is model/get_train_op/learning_rate/learning_rate
        tf.identity(learning_rate, "learning_rate")

        return learning_rate


def record_scalars(metrics_dict):
    for key, value in metrics_dict.items():
        tf.summary.scalar(name=key, tensor=value)
