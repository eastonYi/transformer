from rnnsearch import *


class DeepRNN(RNNSearch):
    def __init__(self, *args, **kargs):
        super(DeepRNN, self).__init__(*args, **kargs)

    def encoder_impl(self, encoder_input, is_training):
        residual_dropout_rate = self._config.residual_dropout_rate if is_training else 0.0

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
        # encoder_output = tf.layers.dropout(encoder_output, rate=residual_dropout_rate, training=is_training)
        encoder_output = common_layers.layer_norm(encoder_output, name='LN_0')

        # Bi-directional RNN
        cell_fw = GRUCell(num_units=self._config.hidden_units,
                          name='fw_cell_0')
        cell_bw = GRUCell(num_units=self._config.hidden_units,
                          name='bw_cell_1')

        encoder_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw, cell_bw=cell_bw,
            inputs=encoder_output,
            sequence_length=sequence_lengths,
            dtype=tf.float32
        )

        encoder_output = tf.concat(encoder_outputs, axis=2)
        encoder_output = dense(encoder_output, output_size=self._config.hidden_units)
        encoder_output = tf.layers.dropout(encoder_output, rate=residual_dropout_rate, training=is_training)

        for i in xrange(2, self._config.num_blocks):
            encoder_output = common_layers.layer_norm(encoder_output, name='LN_%d' % i)

            cell = GRUCell(num_units=self._config.hidden_units,
                           reuse=tf.AUTO_REUSE, name='cell_%s' % i)

            encoder_output_, _ = tf.nn.dynamic_rnn(cell, encoder_output,
                                                   sequence_length=sequence_lengths,
                                                   dtype=tf.float32, scope='rnn_%d' % i)
            encoder_output_ = tf.layers.dropout(encoder_output_, rate=residual_dropout_rate, training=is_training)

            if i >= 2:
                encoder_output = encoder_output_ + encoder_output
            else:
                encoder_output = encoder_output_

        encoder_output = common_layers.layer_norm(encoder_output, name='LN_%d' % self._config.num_blocks)
        # Mask
        encoder_output *= tf.expand_dims(tf.to_float(encoder_mask), axis=-1)

        return encoder_output

    def decoder_impl(self, decoder_input, encoder_output, is_training):

        residual_dropout_rate = self._config.residual_dropout_rate if is_training else 0.0

        attention_bias = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1, keepdims=True), 0.0)
        attention_bias = tf.to_float(attention_bias) * (- 1e9)

        decoder_output = embedding(decoder_input,
                                   vocab_size=self._config.dst_vocab_size,
                                   dense_size=self._config.hidden_units,
                                   kernel=self._dst_embedding,
                                   multiplier=self._config.hidden_units ** 0.5 if self._config.scale_embedding else 1.0,
                                   name="dst_embedding")
        decoder_output = tf.layers.dropout(decoder_output, rate=residual_dropout_rate, training=is_training)

        for i in xrange(self._config.num_blocks):
            decoder_output = common_layers.layer_norm(decoder_output, name='LN_%d' % i)

            if i % 3 == 1:
                cell = AttentionGRUCell(num_units=self._config.hidden_units,
                                        attention_memories=encoder_output,
                                        attention_bias=attention_bias,
                                        reuse=tf.AUTO_REUSE,
                                        name='cell_%s' % i)
            else:
                cell = GRUCell(num_units=self._config.hidden_units,
                               reuse=tf.AUTO_REUSE,
                               name='cell_%s' % i)

            decoder_output_, _ = tf.nn.dynamic_rnn(cell=cell, inputs=decoder_output, dtype=tf.float32,
                                                   scope='rnn_%d' % i)
            decoder_output_ = tf.layers.dropout(decoder_output_, rate=residual_dropout_rate, training=is_training)
            if i >= 2:
                decoder_output = decoder_output_ + decoder_output
            else:
                decoder_output = decoder_output_

        decoder_output = common_layers.layer_norm(decoder_output, name='LN_%d' % self._config.num_blocks)
        return decoder_output

    def decoder_with_caching_impl(self, decoder_input, decoder_cache, encoder_output, is_training):
        residual_dropout_rate = self._config.residual_dropout_rate if is_training else 0.0

        attention_bias = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1, keepdims=True), 0.0)
        attention_bias = tf.to_float(attention_bias) * (- 1e9)

        decoder_input = decoder_input[:, -1]

        decoder_output = embedding(decoder_input,
                                   vocab_size=self._config.dst_vocab_size,
                                   dense_size=self._config.hidden_units,
                                   kernel=self._dst_embedding,
                                   multiplier=self._config.hidden_units ** 0.5 if self._config.scale_embedding else 1.0,
                                   name="dst_embedding")
        decoder_output = tf.layers.dropout(decoder_output, rate=residual_dropout_rate, training=is_training)

        decoder_cache = \
            tf.cond(tf.equal(tf.shape(decoder_cache)[1], 0),
                    lambda: tf.zeros([tf.shape(decoder_input)[0],
                                      1,
                                      self._config.num_blocks,
                                      self._config.hidden_units]),
                    lambda: decoder_cache)
        # Unstack cache
        states = tf.unstack(decoder_cache[:, -1, :, :], num=self._config.num_blocks, axis=1)
        new_states = []
        for i in xrange(self._config.num_blocks):
            decoder_output = common_layers.layer_norm(decoder_output, name='LN_%s' % i)
            if i % 3 == 1:
                cell = AttentionGRUCell(num_units=self._config.hidden_units,
                                        attention_memories=encoder_output,
                                        attention_bias=attention_bias,
                                        reuse=tf.AUTO_REUSE,
                                        name='cell_%s' % i)
            else:
                cell = GRUCell(num_units=self._config.hidden_units,
                               reuse=tf.AUTO_REUSE,
                               name='cell_%s' % i)

            with tf.variable_scope('rnn_%s' % i):
                decoder_output_, decoder_state = cell(decoder_output, states[i])

            # if i % 3 == 1:
            #     # We can log attention weights here.
            #     cell.get_attention_weights()

            decoder_output_ = tf.layers.dropout(decoder_output_, rate=residual_dropout_rate, training=is_training)
            if i >= 2:
                decoder_output = decoder_output_ + decoder_output
            else:
                decoder_output = decoder_output_
            new_states.append(decoder_state)

        decoder_state = tf.stack(new_states, axis=1)[:, None, :, :]
        decoder_output = common_layers.layer_norm(decoder_output, name='LN_%d' % self._config.num_blocks)

        return decoder_output[:, None, :], decoder_state
