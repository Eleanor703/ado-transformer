# -*- coding:utf-8 -*-
"""
@contact: adonis_wu@outlook.com
@file: ado_utils.py
@time: 2018/11/2 09:54
"""
__author__ = '🍊 Adonis Wu 🍊'

import tensorflow as tf
import yaml
from tensorflow.python.estimator.run_config import RunConfig

PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START_TOKEN = '[START]'
STOP_TOKEN = '[STOP]'


def enrich_params(params):
    vocab_path = params.get('vocab_file')
    vocabs = list()
    vocabs.append(PAD_TOKEN)
    vocabs.append(UNKNOWN_TOKEN)
    # vocabs.append(START_TOKEN)
    vocabs.append(STOP_TOKEN)
    data = open(vocab_path, 'r')
    for line in data:
        l = line.split('\t')
        if len(l) == 2:
            vocabs.append(l[0])
            if len(vocabs) == params.get('vocab_size', 50000):
                tf.logging.info('vocab size is {}'.format(params.get('vocab_size', 50000)))
                break
    params['vocabs'] = vocabs


def load_config(path, parent_name):
    """
    :param path: config yaml file path
    :param parent_name: the top setting name
    :return: dict of all settings under the parent name
    """
    yml = yaml.load(open(path, 'r', encoding='utf-8'))
    parameters = yml.get(parent_name)
    for key, value in parameters.items():
        tf.logging.info('{} -> {}'.format(key, value))
    return parameters


def load_sess_config(params):
    """
    :param params: some gpu & session config settings
    :return: run configurations
    """
    if params.get('gpu_cores'):
        #   gpu mode
        tf.logging.warn('using device: {}'.format(params.get('gpu_cores')))
        gpu_options = tf.GPUOptions(
            allow_growth=params.get('allow_growth'),
            visible_device_list=params.get('gpu_cores'),
            per_process_gpu_memory_fraction=params.get('per_process_gpu_memory_fraction')
        )

    else:
        gpu_options = tf.GPUOptions(
            allow_growth=params.get('allow_growth'),
        )
    session_config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=params.get('allow_soft_placement'),

    )
    config = RunConfig(
        save_checkpoints_steps=params.get('save_checkpoints_steps'),
        keep_checkpoint_max=params.get('keep_checkpoint_max'),
        log_step_count_steps=params.get('log_step_count_steps'),
        session_config=session_config

    )
    return config
