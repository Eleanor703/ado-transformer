# -*- coding:utf-8 -*-
"""
@contact: adonis_wu@outlook.com
@file: ado_transformer.py
@time: 2018/11/13 09:18
"""
__author__ = '🍊 Adonis Wu 🍊'

import tensorflow as tf
from model.ado_emb_layer import EmbLayer
from model import model_utils
from model.ado_attention_layer import AttentionLayer,SelfAttention
from model.ado_ffn_layer import FFNLayer
from model import beam_search


class Transformer(object):
    """
    Transformer model for sequence to sequence data
    implemented as described in https://arxiv.org/pdf/1706.03762.pdf
    The transformer model consists of an encoder and decoder.
    The input is an int sequence or a batch of sequences.
    The encoder produces a cotinuous representation, and the decoder uses the encoder output to generate
    probabilities for the output sequence
    """

    def __init__(self, params: dict, train):
        """
        :param params: hyper parameter object defining layer sizes,dropout values, etc.
        :param train:  boolean indicating whether the model is in training mode. Used to
            determine if dropout layers should be added.
        """
        self.params = params
        self.train = train

        self.embedding_layer = EmbLayer(vocab_size=self.params.get('vocab_size'),
                                        hidden_size=self.params.get('hidden_size'))

        self.encoder_stack = EncoderStack(self.params, self.train)
        self.decoder_stack = DecoderStack(self.params, self.train)

    def __call__(self, feature, targets=None):
        """
        :param feature:
        :param targets:
        :return:
        """
        initializer = tf.variance_scaling_initializer(
            scale=self.params.get('initializer_gain'),
            mode='fan_avg',
            distribution='uniform'
        )

        with tf.variable_scope('transformer', initializer=initializer):
            #   [batch_size, 1, 1, length]
            attention_bias = model_utils.get_padding_bias(feature)

            encoder_outputs = self.encode(feature, attention_bias)

            if targets is None:
                return self.predict(encoder_outputs, attention_bias)

            logits = self.decode(targets, encoder_outputs, attention_bias)
            return logits

    def encode(self, inputs, attention_bias):
        """
        :param inputs: int tensor with shape [batch_size, input_length]
        :param attention_bias: float tensor with shape [batch_size, 1, 1, input_length]
        :return: float tensor with shape [batch_size, input_length, hidden_size]
        """
        with tf.name_scope('encode'):
            #   [batch_size, length, hidden_size]
            embedded_inputs = self.embedding_layer(inputs)
            #   [batch_size, length]
            inputs_padding = model_utils.get_padding(inputs)

            with tf.name_scope('add_pos_embedding'):
                length = tf.shape(embedded_inputs)[1]
                #   use sin cos calculate position embeddings
                pos_encoding = model_utils.get_position_encoding(length, self.params.get('hidden_size'))

                encoder_inputs = tf.add(embedded_inputs, pos_encoding)

            if self.train:
                encoder_inputs = tf.nn.dropout(encoder_inputs, 1 - self.params.get('encoder_decoder_dropout'))

            return self.encoder_stack(encoder_inputs, attention_bias, inputs_padding)

    def decode(self, targets, encoder_outputs, attention_bias):
        """
        :param targets:  [batch_size, target_length]
        :param encoder_outputs: [batch_size, input_length, hidden_size]
        :param attention_bias:  [batch_size, 1, 1, input_length]
        :return: [batch_size, target_length, vocab_size]
        """
        with tf.name_scope('decode'):
            #   [batch_size, target_length, hidden_size]
            decoder_inputs = self.embedding_layer(targets)
            with tf.name_scope('shift_targets'):
                #   pad embedding value 0 at the head of sequence and remove eos_id
                decoder_inputs = tf.pad(decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
            with tf.name_scope('add_pos_embedding'):
                length = tf.shape(decoder_inputs)[1]
                position_decode = model_utils.get_position_encoding(length, self.params.get('hidden_size'))
                decoder_inputs = tf.add(decoder_inputs, position_decode)

            if self.train:
                decoder_inputs = tf.nn.dropout(decoder_inputs, 1. - self.params.get('encoder_decoder_dropout'))

            decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(length)

            outputs = self.decoder_stack(
                decoder_inputs,
                encoder_outputs,
                decoder_self_attention_bias,
                attention_bias
            )

            #   [batch_size, target_length, vocab_size]
            logits = self.embedding_layer.linear(outputs)

            return logits

    def predict(self, encoder_outputs, encoder_decoder_attention_bias):
        """
        :param encoder_outputs: [batch_size, input_length, hidden_size]
        :param encoder_decoder_attention_bias: [batch_size, 1, 1, length]
        :return: dict
        """
        batch_size = tf.shape(encoder_outputs)[0]
        max_decode_length = self.params.get('max_decode_length')

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length)

        initial_ids = tf.zeros([batch_size], dtype=tf.int32)

        # Create cache storing decoder attention values for each layer.
        cache = {
            "layer_%d" % layer: {
                "k": tf.zeros([batch_size, 0, self.params["hidden_size"]]),
                "v": tf.zeros([batch_size, 0, self.params["hidden_size"]]),
            } for layer in range(self.params["num_blocks"])}

        # Add encoder output and attention bias to the cache.
        cache["encoder_outputs"] = encoder_outputs
        cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

        #   Top decoded sequences [batch_size, beam_size, max_decode_length]
        #   sequence scores [batch_size, beam_size]
        decoded_ids, scores = beam_search.sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.params.get('vocab_size'),
            beam_size=self.params.get('beam_size'),
            alpha=self.params.get('alpha'),
            max_decode_length=max_decode_length,
            eos_id=self.params.get('eos_id'),
        )

        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        return {"outputs": top_decoded_ids, "scores": top_scores}

    def _get_symbols_to_logits_fn(self, max_decode_length):
        """Returns a decoding function that calculates logits of the next tokens."""

        #   [length, hidden_size]
        timing_signal = model_utils.get_position_encoding(max_decode_length + 1, self.params["hidden_size"])

        #   [1,1,length,length]
        decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(max_decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            """Generate logits for next potential IDs.

            Args:
              ids: Current decoded sequences.
                int tensor with shape [batch_size * beam_size, i + 1]
              i: Loop index
              cache: dictionary of values storing the encoder output, encoder-decoder
                attention bias, and previous decoder attention values.

            Returns:
              Tuple of
                (logits with shape [batch_size * beam_size, vocab_size],
                 updated cache values)
            """
            # Set decoder input to the last generated IDs
            decoder_input = ids[:, -1:]

            # Preprocess decoder input by getting embeddings and adding timing signal.
            decoder_input = self.embedding_softmax_layer(decoder_input)
            decoder_input += timing_signal[i:i + 1]

            self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
            decoder_outputs = self.decoder_stack(
                decoder_input, cache.get("encoder_outputs"), self_attention_bias,
                cache.get("encoder_decoder_attention_bias"), cache)
            logits = self.embedding_softmax_layer.linear(decoder_outputs)
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        return symbols_to_logits_fn


class EncoderStack(tf.layers.Layer):
    def __init__(self, params, train):
        super(EncoderStack, self).__init__()
        self.params = params
        self.train = train

        self.layers = list()
        for _ in range(self.params.get('num_blocks')):
            self_attention_layer = SelfAttention(
                hidden_size=self.params.get('hidden_size'),
                num_heads=self.params.get('num_heads'),
                attention_dropout=self.params.get('attention_dropout'),
                train=self.train
            )

            ffn_layer = FFNLayer(
                hidden_size=self.params.get('hidden_size'),
                filter_size=self.params.get('filter_size'),
                relu_dropout=self.params.get('relu_dropout'),
                train=self.train,
                allow_pad=self.params.get('allow_ffn_pad')
            )

            self.layers.append(
                [
                    PrePostProcessingWrapper(self_attention_layer, self.params, self.train),
                    PrePostProcessingWrapper(ffn_layer, self.params, self.train)
                ]
            )

        self.output_norm = LayerNormalization(self.params.get('hidden_size'))

    def call(self, encoder_inputs, attention_bias, inputs_padding):
        """
        :param encoder_inputs: [batch_size, input_length, hidden_size]
        :param attention_bias: [batch_size, 1, 1, inputs_length]
        :param inputs_padding: [batch_size, length]
        :return: [batch_size, input_length, hidden_size]
        """

        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            ffn_layer = layer[1]
            with tf.variable_scope('encoder_stack_lay_{}'.format(n)):
                with tf.variable_scope('self_attention'):
                    encoder_inputs = self_attention_layer(encoder_inputs, attention_bias)
                with tf.variable_scope('ffn'):
                    encoder_inputs = ffn_layer(encoder_inputs, inputs_padding)

        return self.output_norm(encoder_inputs)


class DecoderStack(tf.layers.Layer):
    def __init__(self, params, train):
        super(DecoderStack, self).__init__()
        self.params = params
        self.train = train
        self.layers = list()
        for _ in range(self.params.get('num_blocks')):
            self_attention_layer = SelfAttention(
                hidden_size=self.params.get('hidden_size'),
                num_heads=self.params.get('num_heads'),
                attention_dropout=self.params.get('attention_dropout'),
                train=self.train
            )

            vanilla_attention_layer = AttentionLayer(
                hidden_size=self.params.get('hidden_size'),
                num_heads=self.params.get('num_heads'),
                attention_dropout=self.params.get('attention_dropout'),
                train=self.train
            )

            ffn_layer = FFNLayer(
                hidden_size=self.params.get('hidden_size'),
                filter_size=self.params.get('filter_size'),
                relu_dropout=self.params.get('relu_dropout'),
                train=self.train,
                allow_pad=self.params.get('allow_ffn_pad')
            )

            self.layers.append(
                [
                    PrePostProcessingWrapper(self_attention_layer, self.params, self.train),
                    PrePostProcessingWrapper(vanilla_attention_layer, self.params, self.train),
                    PrePostProcessingWrapper(ffn_layer, self.params, self.train)
                ]
            )

        self.output_norm = LayerNormalization(self.params.get('hidden_size'))


    def call(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias, attention_bias, cache=None):
        """
        :param decoder_inputs: [batch_size, target_length, hidden_size]
        :param encoder_outputs: [batch_size, input_length, hidden_size]
        :param decoder_self_attention_bias: [1,1,tar]
        :param attention_bias: [batch_size, 1, 1, input_length]
        :return: [batch_size, target_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            vanilla_attention_layer = layer[1]
            ffn_layer = layer[2]

            layer_name = 'layer_{}'.format(n)
            #   for predict
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope('decoder_stack_lay_{}'.format(n)):
                with tf.variable_scope('self_attention'):
                    decoder_inputs = self_attention_layer(decoder_inputs, decoder_self_attention_bias,cache=layer_cache)
                with tf.variable_scope('vanilla_attention'):
                    decoder_inputs = vanilla_attention_layer(decoder_inputs, encoder_outputs, attention_bias)
                with tf.variable_scope('ffn'):
                    decoder_inputs = ffn_layer(decoder_inputs)

        return self.output_norm(decoder_inputs)


class PrePostProcessingWrapper(object):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, layer, params, train):
        self.layer = layer
        self.postprocess_dropout = params["layer_postprocess_dropout"]
        self.train = train

        # Create normalization layer
        self.layer_norm = LayerNormalization(params["hidden_size"])

    def __call__(self, x, *args, **kwargs):
        # Preprocessing: apply layer normalization
        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if self.train:
            y = tf.nn.dropout(y, 1 - self.postprocess_dropout)
        return x + y


class PrePostProcessingWrapper02(tf.layers.Layer):
    def __int__(self, layer, params, train):
        self.layer = layer
        self.params = params
        self.train = train

    def build(self, input_shape):
        self.layer_norm = LayerNormalization(self.params.get('hidden_size'))

        self.built = True

    def call(self, x, *args, **kwargs):
        y = self.layer_norm(x)
        y = self.layer(y, *args, **kwargs)

        if self.train:
            y = tf.nn.dropout(y, 1. - self.params.get('layer_postprocess_dropout'))

        return tf.add(x + y)


class LayerNormalization(tf.layers.Layer):
    """Applies layer normalization."""

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size

    def build(self, _):
        self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                     initializer=tf.ones_initializer())
        self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                    initializer=tf.zeros_initializer())
        self.built = True

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias
