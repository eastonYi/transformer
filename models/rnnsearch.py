from tensorflow.python.ops.rnn_cell import GRUCell
from model import Model
from utils import *


class RNNSearch(Model):
    def __init__(self, *args, **kargs):
        super(RNNSearch, self).__init__(*args, **kargs)
        self._use_daisy_chain_getter = False

    def encoder_impl(self, encoder_input, is_training):
        dropout_rate = self._config.dropout_rate if is_training else 0.0

        # Mask
        encoder_mask = tf.to_int32(tf.not_equal(encoder_input, 0))
        sequence_lengths = tf.reduce_sum(encoder_mask, axis=1)

        # Embedding
        encoder_output = embedding(encoder_input,
                                   vocab_size=self._config.src_vocab_size,
                                   dense_size=self._config.hidden_units,
                                   kernel=self._src_embedding,
                                   multiplier=self._config.hidden_units ** 0.5 if self._config.scale_embedding else 1.0,
                                   name="src_embedding")

        # Dropout
        encoder_output = tf.layers.dropout(encoder_output, rate=dropout_rate, training=is_training)

        cell_fw = GRUCell(num_units=self._config.hidden_units, name='fw_cell')
        cell_bw = GRUCell(num_units=self._config.hidden_units, name='bw_cell')

        # RNN
        encoder_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw, cell_bw=cell_bw,
            inputs=encoder_output,
            sequence_length=sequence_lengths,
            dtype=tf.float32
        )

        encoder_output = tf.concat(encoder_outputs, axis=2)

        # Dropout
        encoder_output = tf.layers.dropout(encoder_output, rate=dropout_rate, training=is_training)

        # Mask
        encoder_output *= tf.expand_dims(tf.to_float(encoder_mask), axis=-1)

        return encoder_output

    def decoder_impl(self, decoder_input, encoder_output, is_training):
        dropout_rate = self._config.dropout_rate if is_training else 0.0

        attention_bias = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1, keepdims=True), 0.0)
        attention_bias = tf.to_float(attention_bias) * (- 1e9)

        decoder_output = embedding(decoder_input,
                                   vocab_size=self._config.dst_vocab_size,
                                   dense_size=self._config.hidden_units,
                                   kernel=self._dst_embedding,
                                   multiplier=self._config.hidden_units ** 0.5 if self._config.scale_embedding else 1.0,
                                   name="dst_embedding")
        decoder_output = tf.layers.dropout(decoder_output, rate=dropout_rate, training=is_training)
        cell = AttentionGRUCell(num_units=self._config.hidden_units,
                                attention_memories=encoder_output,
                                attention_bias=attention_bias,
                                reuse=tf.AUTO_REUSE,
                                name='attention_cell')
        decoder_output, _ = tf.nn.dynamic_rnn(cell=cell, inputs=decoder_output, dtype=tf.float32)
        decoder_output = tf.layers.dropout(decoder_output, rate=dropout_rate, training=is_training)

        return decoder_output

    def decoder_with_caching_impl(self, decoder_input, decoder_cache, encoder_output, is_training):
        dropout_rate = self._config.dropout_rate if is_training else 0.0
        decoder_input = decoder_input[:, -1]
        attention_bias = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1, keepdims=True), 0.0)
        attention_bias = tf.to_float(attention_bias) * (- 1e9)
        decoder_output = embedding(decoder_input,
                                   vocab_size=self._config.dst_vocab_size,
                                   dense_size=self._config.hidden_units,
                                   kernel=self._dst_embedding,
                                   multiplier=self._config.hidden_units ** 0.5 if self._config.scale_embedding else 1.0,
                                   name="dst_embedding")
        cell = AttentionGRUCell(num_units=self._config.hidden_units,
                                attention_memories=encoder_output,
                                attention_bias=attention_bias,
                                reuse=tf.AUTO_REUSE,
                                name='attention_cell')
        decoder_cache = tf.cond(tf.equal(tf.shape(decoder_cache)[1], 0),
                                lambda: tf.zeros([tf.shape(decoder_input)[0], 1, 1, self._config.hidden_units]),
                                lambda: decoder_cache)
        with tf.variable_scope('rnn'):
            decoder_output, _ = cell(decoder_output, decoder_cache[:, -1, -1, :])
        decoder_output = tf.layers.dropout(decoder_output, rate=dropout_rate, training=is_training)
        return decoder_output[:, None, :], decoder_output[:, None, None, :]
