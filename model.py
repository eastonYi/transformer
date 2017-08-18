import tensorflow as tf
from tensorflow.python.ops import init_ops
import logging

import tensor2tensor.common_attention as common_attention
import tensor2tensor.common_layers as common_layers
from utils import FLOAT_TYPE, INT_TYPE, learning_rate_decay, multihead_attention, \
    average_gradients, shift_right, split_tensor, embedding, residual


class Model(object):
    def __init__(self, config):
        self.graph = tf.Graph()
        self.config = config
        self._logger = logging.getLogger('model')
        self._prepared = False

    def prepare(self, is_training):
        assert not self._prepared
        self.is_training = is_training

        # Select devices according to is_training flag.
        devices = self.config.train.devices if is_training else self.config.test.devices
        self.devices = ['/gpu:'+i.strip() for i in devices.split(',') if i] or ['/cpu:0']
        # If we have multiple devices (typically GPUs), we set /cpu:0 as the sync device.
        self._sync_device = self.devices[0] if len(self.devices) == 1 else '/cpu:0'
        if is_training:
            with self.graph.as_default():
                # Optimizer.
                self.global_step = tf.get_variable(name='global_step', dtype=INT_TYPE, shape=[],
                                                   trainable=False, initializer=tf.zeros_initializer)

                if self.config.train.optimizer == 'adam':
                    self.learning_rate = tf.convert_to_tensor(self.config.train.learning_rate)
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                elif self.config.train.optimizer == 'adam_decay':
                    self.learning_rate = learning_rate_decay(self.config, self.global_step)
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                            beta1=0.9, beta2=0.98, epsilon=1e-9)
                elif self.config.train.optimizer == 'sgd':
                    self.learning_rate = tf.convert_to_tensor(self.config.train.learning_rate)
                    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                elif self.config.train.optimizer == 'mom':
                    self.learning_rate = tf.convert_to_tensor(self.config.train.learning_rate)
                    self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)

            # Uniform scaling initializer.
            self._initializer = init_ops.variance_scaling_initializer(scale=1.0, mode='fan_avg', distribution='uniform')

        self._prepared = True

    def build_train_model(self):
        """Build model for training. """
        self.prepare(is_training=True)

        def choose_device(op, device):
            if op.type.startswith('Variable'):
                return self._sync_device
            return device

        with self.graph.as_default(), tf.device(self._sync_device), \
            tf.variable_scope(tf.get_variable_scope(), initializer=self._initializer):
            self.src_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='src_pl')
            self.dst_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='dst_pl')
            Xs = split_tensor(self.src_pl, len(self.devices))
            Ys = split_tensor(self.dst_pl, len(self.devices))
            acc_list, loss_list, gv_list = [], [], []
            for i, (X, Y, device) in enumerate(zip(Xs, Ys, self.devices)):
                with tf.device(lambda op: choose_device(op, device)):
                    self._logger.info('Build model on %s.' % device)
                    encoder_output = self.encoder(X, reuse=i>0 or None)
                    decoder_output = self.decoder(shift_right(Y), encoder_output, reuse=i > 0 or None)
                    acc, loss = self.train_output(decoder_output, Y, reuse=i > 0 or None)
                    acc_list.append(acc)
                    loss_list.append(loss)
                    gv_list.append(self.optimizer.compute_gradients(loss))

            self.acc = tf.reduce_mean(acc_list)
            self.loss = tf.reduce_mean(loss_list)

            # Clip gradients and then apply.
            grads_and_vars = average_gradients(gv_list)
            for g, v in grads_and_vars:
                tf.summary.histogram('variables/' + v.name.split(':')[0], v)
                tf.summary.histogram('gradients/' + v.name.split(':')[0], g)
            grads, self.grads_norm = tf.clip_by_global_norm([gv[0] for gv in grads_and_vars],
                                                            clip_norm=self.config.train.grads_clip)
            grads_and_vars = zip(grads, [gv[1] for gv in grads_and_vars])
            self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

            # Summaries
            tf.summary.scalar('acc', self.acc)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('grads_norm', self.grads_norm)
            self.summary_op = tf.summary.merge_all()

    def build_test_model(self):
        """Build model for inference."""
        self.prepare(is_training=False)

        with self.graph.as_default(), tf.device(self._sync_device):
            self.src_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='src_pl')
            self.dst_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='dst_pl')
            self.decoder_input = shift_right(self.dst_pl)
            Xs = split_tensor(self.src_pl, len(self.devices))
            Ys = split_tensor(self.dst_pl, len(self.devices))
            decoder_inputs = split_tensor(self.decoder_input, len(self.devices))

            prediction_list = []
            loss_sum=0
            for i, (X, Y, decoder_input, device) in enumerate(zip(Xs, Ys, decoder_inputs, self.devices)):
                with tf.device(device):

                    # Avoid errors caused by empty input by a condition phrase.
                    def true_fn():
                        encoder_output = self.encoder(X, reuse=i > 0 or None)
                        prediction = self.beam_search(encoder_output, reuse=i > 0 or None)
                        decoder_output = self.decoder(decoder_input, encoder_output, reuse=True)
                        loss = self.test_loss(decoder_output, Y, reuse=True)
                        return prediction, loss

                    def false_fn():
                        return tf.zeros([0, 0], dtype=INT_TYPE), 0.0

                    prediction, loss = tf.cond(tf.greater(tf.shape(X)[0], 0), true_fn, false_fn)

                    loss_sum += loss
                    prediction_list.append(prediction)

            max_length = tf.reduce_max([tf.shape(pred)[1] for pred in prediction_list])

            def pad_to_max_length(input, length):
                """Pad the input (with rank 2) with 3(</S>) to the given length in the second axis."""
                shape = tf.shape(input)
                padding = tf.ones([shape[0], length - shape[1]], dtype=INT_TYPE) * 3
                return tf.concat([input, padding], axis=1)

            prediction_list = [pad_to_max_length(pred, max_length) for pred in prediction_list]
            self.prediction = tf.concat(prediction_list, axis=0)
            self.loss_sum = loss_sum

    def encoder(self, encoder_input, reuse):
        """Encoder."""
        with tf.variable_scope("encoder", reuse=reuse):
            return self.encoder_body(encoder_input)

    def decoder(self, decoder_input, encoder_output, reuse):
        """Decoder"""
        with tf.variable_scope("decoder", reuse=reuse):
            return self.decoder_body(decoder_input, encoder_output)

    def decoder_with_caching(self, decoder_input, decoder_cache, encoder_output, reuse):
        """Incremental Decoder"""
        with tf.variable_scope("decoder", reuse=reuse):
            return self.decoder_with_caching_body(decoder_input, decoder_cache, encoder_output)

    def beam_search(self, encoder_output, reuse):
        """Beam search in graph."""
        beam_size, batch_size = self.config.test.beam_size, tf.shape(encoder_output)[0]
        inf = 1e10

        def get_bias_scores(scores, bias):
            """
            If a sequence is finished, we only allow one alive branch. This function aims to give one branch a zero score
            and the rest -inf score.
            Args:
                scores: A real value array with shape [batch_size * beam_size, beam_size].
                bias: A bool array with shape [batch_size * beam_size].

            Returns:
                A real value array with shape [batch_size * beam_size, beam_size].
            """
            bias = tf.to_float(bias)
            b = tf.constant([0.0] + [-inf] * (beam_size - 1))
            b = tf.tile(b[None,:], multiples=[batch_size * beam_size, 1])
            return scores * (1 - bias[:, None]) + b * bias[:, None]

        def get_bias_preds(preds, bias):
            """
            If a sequence is finished, all of its branch should be </S> (3).
            Args:
                preds: A int array with shape [batch_size * beam_size, beam_size].
                bias: A bool array with shape [batch_size * beam_size].

            Returns:
                A int array with shape [batch_size * beam_size].
            """
            bias = tf.to_int32(bias)
            return preds * (1 - bias[:, None]) + bias[:, None] * 3

        # Prepare beam search inputs.
        # [batch_size, 1, *, hidden_units]
        encoder_output = encoder_output[:, None, :, :]
        # [batch_size, beam_size, *, hidden_units]
        encoder_output = tf.tile(encoder_output, multiples=[1, beam_size, 1, 1])
        encoder_output = tf.reshape(encoder_output, [batch_size * beam_size, -1, encoder_output.get_shape()[-1].value])
        # [[<S>, <S>, ..., <S>]], shape: [batch_size * beam_size, 1]
        preds = tf.ones([batch_size * beam_size, 1], dtype=INT_TYPE) * 2
        scores = tf.constant([0.0] + [-inf] * (beam_size - 1), dtype=FLOAT_TYPE)  # [beam_size]
        scores = tf.tile(scores, multiples=[batch_size])   # [batch_size * beam_size]
        bias = tf.zeros_like(scores, dtype=tf.bool)
        cache = tf.zeros([batch_size * beam_size, 0, self.config.num_blocks, self.config.hidden_units])

        def step(i, bias, preds, scores, cache):
            # Where are we.
            i += 1

            # Call decoder and get predictions.
            decoder_output, cache = self.decoder_with_caching(preds, cache, encoder_output, reuse=reuse)
            last_preds, last_k_preds, last_k_scores = self.test_output(decoder_output, reuse=reuse)

            last_k_preds = get_bias_preds(last_k_preds, bias)
            last_k_scores = get_bias_scores(last_k_scores, bias)

            # Update scores.
            scores = scores[:, None] + last_k_scores  # [batch_size * beam_size, beam_size]
            scores = tf.reshape(scores, shape=[batch_size, beam_size ** 2])  # [batch_size, beam_size * beam_size]

            # Pruning.
            scores, k_indices = tf.nn.top_k(scores, k=beam_size)
            scores = tf.reshape(scores, shape=[-1])  # [batch_size * beam_size]
            base_indices = tf.reshape(tf.tile(tf.range(batch_size)[:, None], multiples=[1, beam_size]), shape=[-1])
            base_indices *= beam_size ** 2
            k_indices = base_indices + tf.reshape(k_indices, shape=[-1])  # [batch_size * beam_size]

            # Update predictions.
            last_k_preds = tf.gather(tf.reshape(last_k_preds, shape=[-1]), indices=k_indices)
            preds = tf.gather(preds, indices=k_indices/beam_size)
            cache = tf.gather(cache, indices=k_indices/beam_size)
            preds = tf.concat((preds, last_k_preds[:, None]), axis=1)  # [batch_size * beam_size, i]

            # Whether sequences finished.
            bias = tf.equal(preds[:, -1], 3)  # </S>?

            return i, bias, preds, scores, cache

        def not_finished(i, bias, preds, scores, cache):
            return tf.logical_and(
                tf.reduce_any(tf.logical_not(bias)),
                tf.less_equal(
                    i,
                    tf.reduce_min([tf.shape(encoder_output)[1] + 50, self.config.test.max_target_length])
                )
            )

        i, bias, preds, scores, cache = tf.while_loop(cond=not_finished,
                                                      body=step,
                                                      loop_vars=[0, bias, preds, scores, cache],
                                                      shape_invariants=[
                                                          tf.TensorShape([]),
                                                          tf.TensorShape([None]),
                                                          tf.TensorShape([None, None]),
                                                          tf.TensorShape([None]),
                                                          tf.TensorShape([None, None, None, None])],
                                                      back_prop=False)

        scores = tf.reshape(scores, shape=[batch_size, beam_size])
        preds = tf.reshape(preds, shape=[batch_size, beam_size, -1])  # [batch_size, beam_size, max_length]
        lengths = tf.reduce_sum(tf.to_float(tf.not_equal(preds, 3)), axis=-1)   # [batch_size, beam_size]
        lp = tf.pow((5 + lengths) / (5 + 1), self.config.test.lp_alpha)   # Length penalty
        scores /= lp                                                     # following GNMT
        max_indices = tf.to_int32(tf.argmax(scores, axis=-1))   # [batch_size]
        max_indices += tf.range(batch_size) * beam_size
        preds = tf.reshape(preds, shape=[batch_size * beam_size, -1])

        final_preds = tf.gather(preds, indices=max_indices)
        final_preds = final_preds[:, 1:]  # remove <S> flag
        return final_preds

    def test_output(self, decoder_output, reuse):
        """During test, we only need the last prediction at each time."""
        with tf.variable_scope("output", reuse=reuse):
            last_logits = tf.layers.dense(decoder_output[:,-1], self.config.dst_vocab_size)
            last_preds = tf.to_int32(tf.arg_max(last_logits, dimension=-1))
            z = tf.nn.log_softmax(last_logits)
            last_k_scores, last_k_preds = tf.nn.top_k(z, k=self.config.test.beam_size, sorted=False)
            last_k_preds = tf.to_int32(last_k_preds)
        return last_preds, last_k_preds, last_k_scores

    def test_loss(self, decoder_output, Y, reuse):
        """This function help users to compute PPL during test."""
        with tf.variable_scope("output", reuse=reuse):
            logits = tf.layers.dense(decoder_output, self.config.dst_vocab_size)
            mask = tf.to_float(tf.not_equal(Y, 0))
            labels = tf.one_hot(Y, depth=self.config.dst_vocab_size)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss_sum = tf.reduce_sum(loss * mask)
        return loss_sum

    def train_output(self, decoder_output, Y, reuse):
        """Calculate loss and accuracy."""
        with tf.variable_scope("output", reuse=reuse):
            logits = tf.layers.dense(decoder_output, self.config.dst_vocab_size)
            preds = tf.to_int32(tf.arg_max(logits, dimension=-1))
            mask = tf.to_float(tf.not_equal(Y, 0))
            acc = tf.reduce_sum(tf.to_float(tf.equal(preds, Y)) * mask) / tf.reduce_sum(mask)

            # Smoothed loss
            loss = common_layers.smoothing_cross_entropy(logits=logits, labels=Y,
                                                         vocab_size=self.config.dst_vocab_size,
                                                         confidence=1-self.config.train.label_smoothing)
            mean_loss = tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask))

        return acc, mean_loss

    def encoder_body(self, encoder_input):
        """
        This is a interface leave to be implemented by sub classes.
        Args:
            encoder_input: A tensor with shape [batch_size, src_length]

        Returns: A Tensor with shape [batch_size, src_length, num_hidden]

        """
        raise NotImplementedError()

    def decoder_body(self, decoder_input, encoder_output):
        """
        This is a interface leave to be implemented by sub classes.
        Args:
            decoder_input: A Tensor with shape [batch_size, dst_length]
            encoder_output: A Tensor with shape [batch_size, src_length, num_hidden]

        Returns: A Tensor with shape [batch_size, dst_length, num_hidden]

        """
        raise NotImplementedError()

    def decoder_with_caching_body(self, decoder_input, decoder_cache, encoder_output):
        """
        This is a interface leave to be implemented by sub classes.
        Args:
            decoder_input: A Tensor with shape [batch_size, dst_length]
            decoder_cache: A Tensor with shape [batch_size, *, *, num_hidden]
            encoder_output: A Tensor with shape [batch_size, src_length, num_hidden]

        Returns: A Tensor with shape [batch_size, dst_length, num_hidden]

        """
        raise NotImplementedError()


class Transformer(Model):
    def __init__(self, config):
        super(self.__class__, self).__init__(config)

    def encoder_body(self, encoder_input):
        # Mask
        encoder_padding = tf.equal(encoder_input, 0)
        # Embedding
        encoder_output = embedding(encoder_input,
                                   vocab_size=self.config.src_vocab_size,
                                   dense_size=self.config.hidden_units,
                                   multiplier=self.config.hidden_units**0.5 if self.config.scale_embedding else 1.0,
                                   name="src_embedding")
        # Add positional signal
        encoder_output = common_attention.add_timing_signal_1d(encoder_output)
        # Dropout
        encoder_output = tf.layers.dropout(encoder_output,
                                           rate=self.config.residual_dropout_rate,
                                           training=self.is_training)

        # Blocks
        for i in range(self.config.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention
                encoder_output = residual(encoder_output,
                                          common_attention.multihead_attention(
                                              query_antecedent=encoder_output,
                                              memory_antecedent=None,
                                              bias=common_attention.attention_bias_ignore_padding(encoder_padding),
                                              total_key_depth=self.config.hidden_units,
                                              total_value_depth=self.config.hidden_units,
                                              output_depth=self.config.hidden_units,
                                              num_heads=self.config.num_heads,
                                              dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                              name='encoder_self_attention',
                                              summaries=True),
                                          dropout_rate=self.config.residual_dropout_rate,
                                          is_training=self.is_training)

                # Feed Forward
                encoder_output = residual(encoder_output,
                                          common_attention.common_layers.conv_hidden_relu(
                                              inputs=encoder_output,
                                              hidden_size=4 * self.config.hidden_units,
                                              output_size=self.config.hidden_units,
                                              summaries=True),
                                          dropout_rate=self.config.residual_dropout_rate,
                                          is_training=self.is_training)
        # Mask padding part to zeros.
        encoder_output *= tf.expand_dims(1.0 - tf.to_float(encoder_padding), axis=-1)
        return encoder_output

    def decoder_body(self, decoder_input, encoder_output):
        encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
        encoder_attention_bias = common_attention.attention_bias_ignore_padding(encoder_padding)

        decoder_output = embedding(decoder_input,
                                   vocab_size=self.config.dst_vocab_size,
                                   dense_size=self.config.hidden_units,
                                   multiplier=self.config.hidden_units ** 0.5 if self.config.scale_embedding else 1.0,
                                   name="dst_embedding")
        # Positional Encoding
        decoder_output += common_attention.add_timing_signal_1d(decoder_output)
        # Dropout
        decoder_output = tf.layers.dropout(decoder_output,
                                           rate=self.config.residual_dropout_rate,
                                           training=self.is_training)
        # Bias for preventing peeping later information
        self_attention_bias = common_attention.attention_bias_lower_triangle(tf.shape(decoder_input)[1])

        # Blocks
        for i in range(self.config.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention (self-attention)
                decoder_output = residual(decoder_output,
                                          common_attention.multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=None,
                                              bias=self_attention_bias,
                                              total_key_depth=self.config.hidden_units,
                                              total_value_depth=self.config.hidden_units,
                                              num_heads=self.config.num_heads,
                                              dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                              output_depth=self.config.hidden_units,
                                              name="decoder_self_attention",
                                              summaries=True),
                                          dropout_rate=self.config.residual_dropout_rate,
                                          is_training=self.is_training)

                # Multihead Attention (vanilla attention)
                decoder_output = residual(decoder_output,
                                          common_attention.multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=encoder_output,
                                              bias=encoder_attention_bias,
                                              total_key_depth=self.config.hidden_units,
                                              total_value_depth=self.config.hidden_units,
                                              output_depth=self.config.hidden_units,
                                              num_heads=self.config.num_heads,
                                              dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                              name="decoder_vanilla_attention",
                                              summaries=True),
                                          dropout_rate=self.config.residual_dropout_rate,
                                          is_training=self.is_training)

                # Feed Forward
                decoder_output = residual(decoder_output,
                                          common_layers.conv_hidden_relu(
                                              decoder_output,
                                              hidden_size=4 * self.config.hidden_units,
                                              output_size=self.config.hidden_units,
                                              summaries=True),
                                          dropout_rate=self.config.residual_dropout_rate,
                                          is_training=self.is_training)
        return decoder_output

    def decoder_with_caching_body(self, decoder_input, decoder_cache, encoder_output):
        encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
        encoder_attention_bias = common_attention.attention_bias_ignore_padding(encoder_padding)

        decoder_output = embedding(decoder_input,
                                   vocab_size=self.config.dst_vocab_size,
                                   dense_size=self.config.hidden_units,
                                   multiplier=self.config.hidden_units ** 0.5 if self.config.scale_embedding else 1.0,
                                   name="dst_embedding")
        # Positional Encoding
        decoder_output += common_attention.add_timing_signal_1d(decoder_output)
        # Dropout
        decoder_output = tf.layers.dropout(decoder_output,
                                           rate=self.config.residual_dropout_rate,
                                           training=self.is_training)

        new_cache = []

        # Blocks
        for i in range(self.config.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention (self-attention)
                decoder_output = residual(decoder_output[:, -1:, :],
                                          multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=None,
                                              bias=None,
                                              total_key_depth=self.config.hidden_units,
                                              total_value_depth=self.config.hidden_units,
                                              num_heads=self.config.num_heads,
                                              dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                              reserve_last=True,
                                              output_depth=self.config.hidden_units,
                                              name="decoder_self_attention",
                                              summaries=True),
                                          dropout_rate=self.config.residual_dropout_rate,
                                          is_training=self.is_training)

                # Multihead Attention (vanilla attention)
                decoder_output = residual(decoder_output,
                                          multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=encoder_output,
                                              bias=encoder_attention_bias,
                                              total_key_depth=self.config.hidden_units,
                                              total_value_depth=self.config.hidden_units,
                                              output_depth=self.config.hidden_units,
                                              num_heads=self.config.num_heads,
                                              dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                              reserve_last=True,
                                              name="decoder_vanilla_attention",
                                              summaries=True),
                                          dropout_rate=self.config.residual_dropout_rate,
                                          is_training=self.is_training)

                # Feed Forward
                decoder_output = residual(decoder_output,
                                          common_layers.conv_hidden_relu(
                                              decoder_output,
                                              hidden_size=4 * self.config.hidden_units,
                                              output_size=self.config.hidden_units,
                                              summaries=True),
                                          dropout_rate=self.config.residual_dropout_rate,
                                          is_training=self.is_training)

                decoder_output = tf.concat([decoder_cache[:, :, i, :], decoder_output], axis=1)
                new_cache.append(decoder_output[:, :, None, :])

        new_cache = tf.concat(new_cache, axis=2)  # [batch_size, n_step, num_blocks, num_hidden]

        return decoder_output, new_cache
