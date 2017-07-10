import tensorflow as tf
from tensorflow.python.ops import init_ops
import numpy as np
import logging

from tensor2tensor.common_attention import multihead_attention, add_timing_signal_1d, attention_bias_ignore_padding, attention_bias_lower_triangle
from tensor2tensor.common_layers import layer_norm, embedding, conv_hidden_relu, smoothing_cross_entropy

INT_TYPE = np.int32
FLOAT_TYPE = np.float32


class Model(object):
    def __init__(self, config):
        self.graph = tf.Graph()
        self.config = config
        self._logger = logging.getLogger('model')
        self._prepared = False

    def prepare(self, is_training):
        assert not self._prepared
        self.is_training = is_training
        # Select devices according to running is_training flag.
        devices = self.config.train.devices if is_training else self.config.test.devices
        self.devices = ['/gpu:'+i for i in devices.split(',')] or ['/cpu:0']
        # If we have multiple devices (typically GPUs), we set /cpu:0 as the sync device.
        self.sync_device = self.devices[0] if len(self.devices) == 1 else '/cpu:0'

        if is_training:
            with self.graph.as_default():
                with tf.device(self.sync_device):
                    # Preparing optimizer.
                    self.global_step = tf.get_variable(name='global_step', dtype=INT_TYPE, shape=[],
                                                       trainable=False, initializer=tf.zeros_initializer)
                    self.learning_rate = self.config.train.learning_rate * learning_rate_decay(self.config, self.global_step)
                    if self.config.train.optimizer == 'normal_adam':
                        self.learning_rate = tf.convert_to_tensor(self.config.train.learning_rate)
                        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                    elif self.config.train.optimizer == 'adam':
                        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9)
                    elif self.config.train.optimizer == 'sgd':
                        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                    elif self.config.train.optimizer == 'mom':
                        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
        self._initializer = init_ops.variance_scaling_initializer(scale=1, mode='fan_avg', distribution='uniform')
        self._prepared = True

    def build_train_model(self):
        """Build model for training. """

        self.prepare(is_training=True)

        with self.graph.as_default():
            with tf.device(self.sync_device):
                self.src_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='src_pl')
                self.dst_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='dst_pl')
                Xs = split_tensor(self.src_pl, len(self.devices))
                Ys = split_tensor(self.dst_pl, len(self.devices))
                acc_list, loss_list, gv_list = [], [], []
                for i, (X, Y, device) in enumerate(zip(Xs, Ys, self.devices)):
                    with tf.device(lambda op: self.choose_device(op, device)):
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
                grads, self.grads_norm = tf.clip_by_global_norm([gv[0] for gv in grads_and_vars],
                                                                clip_norm=self.config.train.grads_clip)
                grads_and_vars = zip(grads, [gv[1] for gv in grads_and_vars])
                for g, v in grads_and_vars:
                    tf.summary.histogram('gradients of ' + v.name, g)
                self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

                # Summaries
                tf.summary.scalar('acc', self.acc)
                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar('learning_rate', self.learning_rate)
                tf.summary.scalar('grads_norm', self.grads_norm)
                self.summary_op = tf.summary.merge_all()

    def build_test_model(self):
        """Build model for testing."""

        self.prepare(is_training=False)

        with self.graph.as_default():
            with tf.device(self.sync_device):
                self.src_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='src_pl')
                self.dst_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='dst_pl')
                self.decoder_input = shift_right(self.dst_pl)
                Xs = split_tensor(self.src_pl, len(self.devices))
                Ys = split_tensor(self.dst_pl, len(self.devices))
                dec_inputs = split_tensor(self.decoder_input, len(self.devices))

                # Encode
                encoder_output_list = []
                for i, (X, device) in enumerate(zip(Xs, self.devices)):
                    with tf.device(lambda op: self.choose_device(op, device)):
                        encoder_output = self.encoder(X, reuse=i > 0 or None)
                        encoder_output_list.append(encoder_output)
                self.encoder_output = tf.concat(encoder_output_list, axis=0)

                # Decode
                enc_outputs = split_tensor(self.encoder_output, len(self.devices))
                preds_list, k_preds_list, k_scores_list = [], [], []
                self.loss_sum = 0.0
                for i, (X, enc_output, dec_input, Y, device) in enumerate(zip(Xs, enc_outputs, dec_inputs, Ys, self.devices)):
                    with tf.device(lambda op: self.choose_device(op, device)):
                        self._logger.info('Build model on %s.' % device)
                        decoder_output = self.decoder(dec_input, enc_output, reuse=i > 0 or None)
                        # Predictions
                        preds, k_preds, k_scores = self.test_output(decoder_output, reuse=i > 0 or None)
                        preds_list.append(preds)
                        k_preds_list.append(k_preds)
                        k_scores_list.append(k_scores)
                        # Loss
                        loss = self.test_loss(decoder_output, Y, reuse=True)
                        self.loss_sum += loss

                self.preds = tf.concat(preds_list, axis=0)
                self.k_preds = tf.concat(k_preds_list, axis=0)
                self.k_scores = tf.concat(k_scores_list, axis=0)

    def choose_device(self, op, device):
        """Choose a device according the op's type."""
        if op.type.startswith('Variable'):
            return self.sync_device
        return device

    def encoder(self, encoder_input, reuse):
        """Transformer encoder."""
        with tf.variable_scope("encoder", initializer=self._initializer, reuse=reuse):
            # Mask
            encoder_padding = tf.equal(encoder_input, 0)
            # Embedding
            encoder_output = embedding(encoder_input,
                                       vocab_size=self.config.src_vocab_size,
                                       dense_size=self.config.hidden_units,
                                       multiplier=self.config.hidden_units**0.5,
                                       name="src_embedding")
            # Add positional signal
            encoder_output = add_timing_signal_1d(encoder_output)
            # Dropout
            encoder_output = tf.layers.dropout(encoder_output,
                                               rate=self.config.residual_dropout_rate,
                                               training=self.is_training)

            # Blocks
            for i in range(self.config.num_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    # Multihead Attention
                    encoder_output = residual(encoder_output,
                                              multihead_attention(
                                                  query_antecedent=encoder_output,
                                                  memory_antecedent=None,
                                                  bias=attention_bias_ignore_padding(encoder_padding),
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
                                              conv_hidden_relu(
                                                  inputs=encoder_output,
                                                  hidden_size=4 * self.config.hidden_units,
                                                  output_size=self.config.hidden_units),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)
        # Mask padding part to zeros.
        encoder_output *= tf.expand_dims(1.0 - tf.to_float(encoder_padding), axis=-1)
        return encoder_output

    def decoder(self, decoder_input, encoder_output, reuse):
        """Transformer decoder"""
        with tf.variable_scope("decoder", initializer=self._initializer, reuse=reuse):
            encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
            encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)

            decoder_output = embedding(decoder_input,
                                       vocab_size=self.config.dst_vocab_size,
                                       dense_size=self.config.hidden_units,
                                       multiplier=self.config.hidden_units**0.5,
                                       name="dst_embedding")
            # Positional Encoding
            decoder_output += add_timing_signal_1d(decoder_output)
            # Dropout
            decoder_output = tf.layers.dropout(decoder_output,
                                               rate=self.config.residual_dropout_rate,
                                               training=self.is_training)
            # Bias for preventing peeping later information
            self_attention_bias = attention_bias_lower_triangle(tf.shape(decoder_input)[1])

            # Blocks
            for i in range(self.config.num_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    # Multihead Attention (self-attention)
                    decoder_output = residual(decoder_output,
                                              multihead_attention(
                                                  query_antecedent=decoder_output,
                                                  memory_antecedent=None,
                                                  bias=self_attention_bias,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0,
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
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0,
                                                  name="decoder_vanilla_attention",
                                                  summaries=True),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

                    # Feed Forward
                    decoder_output = residual(decoder_output,
                                              conv_hidden_relu(
                                                  decoder_output,
                                                  hidden_size=4 * self.config.hidden_units,
                                                  output_size=self.config.hidden_units),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

            return decoder_output

    def test_output(self, decoder_output, reuse):
        """During test, we only need the last prediction."""
        with tf.variable_scope("output", reuse=reuse):
            last_logits = tf.layers.dense(decoder_output[:,-1], self.config.dst_vocab_size)
            last_preds = tf.to_int32(tf.arg_max(last_logits, dimension=-1))
            z = tf.nn.log_softmax(last_logits)
            last_k_scores, last_k_preds = tf.nn.top_k(z, k=self.config.test.beam_size, sorted=False)
            last_k_preds = tf.to_int32(last_k_preds)
        return last_preds, last_k_preds, last_k_scores

    def test_loss(self, decoder_output, Y, reuse):
        with tf.variable_scope("output", initializer=self._initializer, reuse=reuse):
            logits = tf.layers.dense(decoder_output, self.config.dst_vocab_size)
            mask = tf.to_float(tf.not_equal(Y, 0))
            labels = tf.one_hot(Y, depth=self.config.dst_vocab_size)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss_sum = tf.reduce_sum(loss * mask)
        return loss_sum

    def train_output(self, decoder_output, Y, reuse):
        """Calculate loss and accuracy."""
        with tf.variable_scope("output", initializer=self._initializer, reuse=reuse):
            logits = tf.layers.dense(decoder_output, self.config.dst_vocab_size)
            preds = tf.to_int32(tf.arg_max(logits, dimension=-1))
            mask = tf.to_float(tf.not_equal(Y, 0))
            acc = tf.reduce_sum(tf.to_float(tf.equal(preds, Y)) * mask) / tf.reduce_sum(mask)

            # Smoothed loss
            loss = smoothing_cross_entropy(logits=logits, labels=Y, vocab_size=self.config.dst_vocab_size,
                                           confidence=1-self.config.train.label_smoothing)
            mean_loss = tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask))

        return acc, mean_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        else:
            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads


def residual(inputs, outputs, dropout_rate, is_training):
    """Residual connection.

    Args:
        inputs: A Tensor.
        outputs: A Tensor.
        dropout_rate: A float.
        is_training: A bool.

    Returns:
        A Tensor.
    """
    output = inputs + tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)
    output = layer_norm(output)
    return output


def split_tensor(input, n):
    """
    Split the tensor input to n tensors.
    Args:
        inputs: A tensor with size [b, ...].
        n: A integer.

    Returns: A tensor list, each tensor has size [b/n, ...].
    """
    batch_size = tf.shape(input)[0]
    ls = tf.cast(tf.lin_space(0.0, tf.cast(batch_size, FLOAT_TYPE), n + 1), INT_TYPE)
    return [input[ls[i]:ls[i+1]] for i in range(n)]


def learning_rate_decay(config, global_step):
    """Inverse-decay learning rate until warmup_steps, then decay."""
    warmup_steps = tf.to_float(config.train.learning_rate_warmup_steps)
    global_step = tf.to_float(global_step)
    return config.hidden_units ** -0.5 * tf.minimum(
        (global_step + 1.0) * warmup_steps ** -1.5, (global_step + 1.0) ** -0.5)


def shift_right(input, pad=2):
    """Shift input tensor right to create decoder input. '2' denotes <S>"""
    return tf.concat((tf.ones_like(input[:, :1]) * pad, input[:, :-1]), 1)
