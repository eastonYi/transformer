from __future__ import print_function
import numpy as np
import os
import codecs
import logging
from tempfile import mkstemp
from itertools import izip
import tensorflow as tf

from tensor2tensor import common_layers, common_attention

INT_TYPE = np.int32
FLOAT_TYPE = np.float32


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]


class DataUtil(object):
    """
    Util class for creating batches for training and testing.
    """

    def __init__(self, config):
        self.config = config
        self._logger = logging.getLogger('util')
        self._tmps = set()
        self.load_vocab()

    def __del__(self):
        for fname in self._tmps:
            if os.path.exists(fname):
                os.remove(fname)

    def load_vocab(self):
        """
        Load vocab from disk. The fisrt four items in the vocab should be <PAD>, <UNK>, <S>, </S>
        """

        def load_vocab_(path, vocab_size):
            vocab = [line.split()[0] for line in codecs.open(path, 'r', 'utf-8')]
            vocab = vocab[:vocab_size]
            assert len(vocab) == vocab_size
            word2idx = {word: idx for idx, word in enumerate(vocab)}
            idx2word = {idx: word for idx, word in enumerate(vocab)}
            return word2idx, idx2word

        self._logger.debug('Load vocabularies %s and %s.' % (self.config.src_vocab, self.config.dst_vocab))
        self.src2idx, self.idx2src = load_vocab_(self.config.src_vocab, self.config.src_vocab_size)
        self.dst2idx, self.idx2dst = load_vocab_(self.config.dst_vocab, self.config.dst_vocab_size)

    def get_training_batches(self, shuffle=True):
        """
        Generate batches with fixed batch size.
        """

        src_path = self.config.train.src_path
        dst_path = self.config.train.dst_path
        batch_size = self.config.train.batch_size
        max_length = self.config.train.max_length

        # Shuffle the training files.
        if shuffle:
            src_shuf_path, dst_shuf_path = self.shuffle([src_path, dst_path])
        else:
            src_shuf_path = src_path
            dst_shuf_path = dst_path

        src_sents, dst_sents = [], []
        for src_sent, dst_sent in izip(open(src_shuf_path, 'r'), open(dst_shuf_path, 'r')):
            src_sent, dst_sent = src_sent.decode('utf8'), dst_sent.decode('utf8')
            # If exceed the max length, abandon this sentence pair.
            src_sent = src_sent.split()
            dst_sent = dst_sent.split()
            if  len(src_sent) > max_length or len(dst_sent) > max_length:
                continue
            src_sents.append(src_sent)
            dst_sents.append(dst_sent)
            # Create a padded batch.
            if len(src_sents) >= batch_size:
                yield self.create_batch(src_sents, o='src'), self.create_batch(dst_sents, o='dst')
                src_sents, dst_sents = [], []

        if src_sents and dst_sents:
            yield self.create_batch(src_sents, o='src'), self.create_batch(dst_sents, o='dst')

        # Remove shuffled files when epoch finished.
        if shuffle:
            os.remove(src_shuf_path)
            os.remove(dst_shuf_path)

    def get_training_batches_with_buckets(self, shuffle=True):
        """
        Generate batches according to bucket setting.
        """

        buckets = [(i, i) for i in range(10, 100, 5)] + [(self.config.train.max_length, self.config.train.max_length)]

        def select_bucket(sl, dl):
            for l1, l2 in buckets:
                if sl < l1 and dl < l2:
                    return (l1, l2)
            return None

        # Shuffle the training files.
        src_path = self.config.train.src_path
        dst_path = self.config.train.dst_path
        if shuffle:
            self._logger.debug('Shuffle files %s and %s.' % (src_path, dst_path))
            src_shuf_path, dst_shuf_path = self.shuffle([src_path, dst_path])
            self._tmps.add(src_shuf_path)
            self._tmps.add(dst_shuf_path)
        else:
            src_shuf_path = src_path
            dst_shuf_path = dst_path

        caches = {}
        for bucket in buckets:
            caches[bucket] = [[], [], 0, 0]  # src sentences, dst sentences, src tokens, dst tokens

        for src_sent, dst_sent in izip(open(src_shuf_path, 'r'), open(dst_shuf_path, 'r')):
            src_sent, dst_sent = src_sent.decode('utf8'), dst_sent.decode('utf8')

            src_sent = src_sent.split()
            dst_sent = dst_sent.split()

            bucket = select_bucket(len(src_sent), len(dst_sent))
            if bucket is None:  # No bucket is selected when the sentence length exceed the max length.
                continue

            caches[bucket][0].append(src_sent)
            caches[bucket][1].append(dst_sent)
            caches[bucket][2] += len(src_sent)
            caches[bucket][3] += len(dst_sent)

            if max(caches[bucket][2], caches[bucket][3]) >= self.config.train.tokens_per_batch:
                batch = (self.create_batch(caches[bucket][0], o='src'), self.create_batch(caches[bucket][1], o='dst'))
                self._logger.debug(
                    'Yield batch with source shape %s and target shape %s.' % (batch[0].shape, batch[1].shape))
                yield batch
                caches[bucket] = [[], [], 0, 0]

        # Clean remain sentences.
        for bucket in buckets:
            # Ensure each device at least get one sample.
            if len(caches[bucket][0]) > len(self.config.train.devices.split(',')):
                batch = (self.create_batch(caches[bucket][0], o='src'), self.create_batch(caches[bucket][1], o='dst'))
            self._logger.debug(
                'Yield batch with source shape %s and target shape %s.' % (batch[0].shape, batch[1].shape))
            yield batch

        # Remove shuffled files when epoch finished.
        if shuffle:
            os.remove(src_shuf_path)
            os.remove(dst_shuf_path)
            self._tmps.remove(src_shuf_path)
            self._tmps.remove(dst_shuf_path)

    @staticmethod
    def shuffle(list_of_files):
        tf_os, tpath = mkstemp()
        tf = open(tpath, 'w')

        fds = [open(ff) for ff in list_of_files]

        for l in fds[0]:
            lines = [l.strip()] + [ff.readline().strip() for ff in fds[1:]]
            print("|||||".join(lines), file=tf)

        [ff.close() for ff in fds]
        tf.close()

        os.system('shuf %s > %s' % (tpath, tpath + '.shuf'))

        fds = [open(ff + '.{}.shuf'.format(os.getpid()), 'w') for ff in list_of_files]

        for l in open(tpath + '.shuf'):
            s = l.strip().split('|||||')
            for i, fd in enumerate(fds):
                print(s[i], file=fd)

        [ff.close() for ff in fds]

        os.remove(tpath)
        os.remove(tpath + '.shuf')

        return [ff + '.{}.shuf'.format(os.getpid()) for ff in list_of_files]

    def get_test_batches(self):
        src_path = self.config.test.src_path
        batch_size = self.config.test.batch_size

        # Read batches from test files.
        src_sents = []
        for src_sent in open(src_path, 'r'):
            src_sent = src_sent.decode('utf8')
            src_sent = src_sent.split()
            src_sents.append(src_sent)
            # Create a padded batch.
            if len(src_sents) >= batch_size:
                yield self.create_batch(src_sents, o='src')
                src_sents = []
        if src_sents:
            yield self.create_batch(src_sents, o='src')

    def get_test_batches_with_target(self):
        """
        Usually we don't need target sentences for test unless we want to compute PPl.
        Returns:
            Paired source and target batches.
        """
        src_path = self.config.test.src_path
        dst_path = self.config.test.dst_path
        batch_size = self.config.test.batch_size

        # Read batches from test files.
        src_sents, dst_sents = [], []
        for src_sent, dst_sent in izip(open(src_path, 'r'), open(dst_path, 'r')):
            src_sent, dst_sent = src_sent.decode('utf8'), dst_sent.decode('utf8')
            src_sent = src_sent.split()
            dst_sent = dst_sent.split()
            src_sents.append(src_sent)
            dst_sents.append(dst_sent)
            # Create a padded batch.
            if len(src_sents) >= batch_size:
                yield self.create_batch(src_sents, o='src'), self.create_batch(dst_sents, o='dst')
                src_sents, dst_sents = [], []
        if src_sents:
            yield self.create_batch(src_sents, o='src'), self.create_batch(dst_sents, o='dst')

    def create_batch(self, sents, o):
        # Convert words to indices.
        assert o in ('src', 'dst')
        word2idx = self.src2idx if o == 'src' else self.dst2idx
        indices = []
        for sent in sents:
            x = [word2idx.get(word, 1) for word in (sent + [u"</S>"])]  # 1: OOV, </S>: End of Text
            indices.append(x)

        # Pad to the same length.
        maxlen = max([len(s) for s in indices])
        X = np.zeros([len(indices), maxlen], np.int32)
        for i, x in enumerate(indices):
            X[i, :len(x)] = x

        return X

    def indices_to_words(self, Y, o='dst'):
        assert o in ('src', 'dst')
        idx2word = self.idx2src if o == 'src' else self.idx2dst
        sents = []
        for y in Y: # for each sentence
            sent = []
            for i in y: # For each word
                if i == 3:  # </S>
                    break
                w = idx2word[i]
                sent.append(w)
            sents.append(' '.join(sent))
        return sents


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
    output = common_layers.layer_norm(output)
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


def embedding(x, vocab_size, dense_size, name=None, reuse=None, multiplier=1.0):
    """Embed x of type int64 into dense vectors."""
    with tf.variable_scope(
        name, default_name="embedding", values=[x], reuse=reuse):
        embedding_var = tf.get_variable("kernel", [vocab_size, dense_size])
        emb_x = tf.gather(embedding_var, x)
        if multiplier != 1.0:
            emb_x *= multiplier
        return emb_x


def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        reserve_last=False,
                        summaries=False,
                        image_shapes=None,
                        name=None):
    """Multihead scaled-dot-product attention with input/output transformations.

    Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    reserve_last: a boolean
    summaries: a boolean
    image_shapes: optional quadruple of integer scalars for image summary.
        If the query positions and memory positions represent the
        pixels of a flattened image, then pass in their dimensions:
          (query_rows, query_cols, memory_rows, memory_cols).
    name: an optional string

    Returns:
    A Tensor.
    """
    with tf.variable_scope(
        name,
        default_name="multihead_attention",
        values=[query_antecedent, memory_antecedent]):
        if memory_antecedent is None:
            # self attention
            combined = common_layers.conv1d(
              query_antecedent,
              total_key_depth * 2 + total_value_depth,
              1,
              name="qkv_transform")
            q, k, v = tf.split(
              combined, [total_key_depth, total_key_depth, total_value_depth],
              axis=2)
        else:
            q = common_layers.conv1d(
              query_antecedent, total_key_depth, 1, name="q_transform")
            combined = common_layers.conv1d(
              memory_antecedent,
              total_key_depth + total_value_depth,
              1,
              name="kv_transform")
            k, v = tf.split(combined, [total_key_depth, total_value_depth], axis=2)

        if reserve_last:
            q = q[:, -1:, :]

        q = common_attention.split_heads(q, num_heads)
        k = common_attention.split_heads(k, num_heads)
        v = common_attention.split_heads(v, num_heads)
        key_depth_per_head = total_key_depth // num_heads
        q *= key_depth_per_head**-0.5
        x = common_attention.dot_product_attention(
            q, k, v, bias, dropout_rate, summaries, image_shapes)
        x = common_attention.combine_heads(x)
        x = common_layers.conv1d(x, output_depth, 1, name="output_transform")
        return x