from __future__ import print_function
import numpy as np
import os
import codecs
from tempfile import mkstemp
from itertools import izip
import tensorflow as tf


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
        self.load_vocab()

    def load_vocab(self):

        def load_vocab_(path, vocab_size):
            vocab = [line.split()[0] for line in codecs.open(path, 'r', 'utf-8')]
            vocab = vocab[:vocab_size]
            assert len(vocab) == vocab_size
            word2idx = {word: idx for idx, word in enumerate(vocab)}
            idx2word = {idx: word for idx, word in enumerate(vocab)}
            return word2idx, idx2word

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
        for src_sent, dst_sent in izip(codecs.open(src_shuf_path, 'r', 'utf8'),
                                       codecs.open(dst_shuf_path, 'r', 'utf8')):
            # If exceed the max length, abandon this sentence pair.
            src_sent = src_sent.split()
            dst_sent = dst_sent.split()
            if  len(src_sent) > max_length or len(dst_sent) > max_length:
                continue
            src_sents.append(src_sent)
            dst_sents.append(dst_sent)
            # Create a padded batch.
            if len(src_sents) >= batch_size:
                yield self.create_batch(src_sents), self.create_batch(dst_sents)
                src_sents, dst_sents = [], []

        if src_sents and dst_sents:
            yield self.create_batch(src_sents), self.create_batch(dst_sents)

        # Remove shuffled files when epoch finished.
        if shuffle:
            os.remove(src_shuf_path)
            os.remove(dst_shuf_path)

    def get_training_batches_with_buckets(self, shuffle=True):
        """
        Generate batches according to bucket setting.
        """

        buckets = [(i, i) for i in range(10, 100, 10)] + [(self.config.train.max_length, self.config.train.max_length)]

        def select_bucket(sl, dl):
            for l1, l2 in buckets:
                if sl < l1 and dl < l2:
                    return (l1, l2)
            return None

        # Shuffle the training files.
        src_path = self.config.train.src_path
        dst_path = self.config.train.dst_path
        if shuffle:
            src_shuf_path, dst_shuf_path = self.shuffle([src_path, dst_path])
        else:
            src_shuf_path = src_path
            dst_shuf_path = dst_path

        caches = {}
        for bucket in buckets:
            caches[bucket] = [[], [], 0, 0]  # src sentences, dst sentences, src tokens, dst tokens

        for src_sent, dst_sent in izip(codecs.open(src_shuf_path, 'r', 'utf8'),
                                       codecs.open(dst_shuf_path, 'r', 'utf8')):
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
                yield self.create_batch(caches[bucket][0]), self.create_batch(caches[bucket][1])
                caches[bucket] = [[], [], 0, 0]
                # TODO
                return
        # Clean remain sentences
        for bucket in buckets:
            if len(caches[bucket][0]) > 10:  # If there are more than 20 sentences in the bucket.
                yield self.create_batch(caches[bucket][0]), self.create_batch(caches[bucket][1])

        # Remove shuffled files when epoch finished.
        if shuffle:
            os.remove(src_shuf_path)
            os.remove(dst_shuf_path)

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

        fds = [open(ff + '.shuf', 'w') for ff in list_of_files]

        for l in open(tpath + '.shuf'):
            s = l.strip().split('|||||')
            for i, fd in enumerate(fds):
                print(s[i], file=fd)

        [ff.close() for ff in fds]

        os.remove(tpath)
        os.remove(tpath + '.shuf')

        return [ff + '.shuf' for ff in list_of_files]

    def get_test_batches(self):
        src_path = self.config.test.src_path
        batch_size = self.config.test.batch_size

        # Read batches from test files.
        src_sents = []
        for src_sent in codecs.open(src_path, 'r', 'utf8'):
            src_sent = src_sent.split()
            src_sents.append(src_sent)
            # Create a padded batch.
            if len(src_sents) >= batch_size:
                yield self.create_batch(src_sents)
                src_sents = []
        if src_sents:
            yield self.create_batch(src_sents)

    def create_batch(self, sents):
        # Convert words to indices.
        indices = []
        for sent in sents:
            x = [self.src2idx.get(word, 1) for word in (sent + [u"</S>"])]  # 1: OOV, </S>: End of Text
            indices.append(x)

        # Pad to the same length.
        maxlen = max([len(s) for s in indices])
        X = np.zeros([len(indices), maxlen], np.int32)
        for i, x in enumerate(indices):
            X[i, :len(x)] = x

        return X

    def indices_to_words(self, Y):
        sents = []
        for y in Y: # for each sentence
            sent = []
            for i in y: # For each word
                if i == 3:  # </S>
                    break
                w = self.idx2dst[i]
                sent.append(w)
            sents.append(' '.join(sent))
        return sents
