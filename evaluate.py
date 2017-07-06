from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np
import yaml
from tempfile import mkstemp
from argparse import ArgumentParser

from model import Model, INT_TYPE
from utils import DataUtil, AttrDict


class Evaluator(object):
    """
    Evaluate the model.
    """
    def __init__(self, config):
        self.config = config

        # Load model
        self.model = Model(config)
        self.model.build_test_model()

        # Create session
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.allow_soft_placement = True
        self.sess = tf.Session(config=sess_config, graph=self.model.graph)
        # Restore model.
        with self.model.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, tf.train.latest_checkpoint(config.train.logdir))

    def __del__(self):
        self.sess.close()

    def greedy_search(self, X):
        """
        Greedy search.
        Args:
            X: A 2-d array with size [n, src_length], source sentence indices.

        Returns:
            A 2-d array with size [n, dst_length], destination sentence indices.
        """
        encoder_output = self.sess.run(self.model.encoder_output, feed_dict={self.model.src_pl: X})
        preds = np.ones([X.shape[0], 1], dtype=INT_TYPE) * 2 # <S>
        finish = np.zeros(X.shape[0:1], dtype=np.bool)
        for i in xrange(config.test.max_length):
            last_preds = self.sess.run(self.model.preds, feed_dict={self.model.encoder_output: encoder_output,
                                                                    self.model.decoder_input: preds})
            finish += last_preds == 3   # </S>
            if finish.all():
                break
            preds = np.concatenate((preds, last_preds[:, None]), axis=1)

        return preds[:, 1:]

    def beam_search(self, X):
        """
        Beam search.
        Args:
            X: A 2-d array with size [n, src_length], source sentence indices.

        Returns:
            A 2-d array with size [n, dst_length], destination sentence indices.
        """
        beam_size, batch_size = config.test.beam_size, X.shape[0]
        encoder_output = self.sess.run(self.model.encoder_output, feed_dict={self.model.src_pl: X})
        preds = np.ones([batch_size, 1], dtype=INT_TYPE) * 2  # <S>
        last_k_preds, last_k_scores = self.sess.run([self.model.k_preds, self.model.k_scores],
                                                    feed_dict={self.model.encoder_output: encoder_output,
                                                               self.model.decoder_input: preds})  # [batch_size, beam_size]

        preds = np.concatenate((np.ones([batch_size * beam_size, 1], dtype=INT_TYPE) * 2, last_k_preds.flatten()), 1)   # [batch_size * beam_size, 2]
        scores = last_k_scores.flatten()   # [batch_size * beam_size]
        encoder_output = np.array(zip(*[encoder_output]*beam_size)).reshape([batch_size * beam_size, -1])  # [batch_size * beam_size, hidden_units]

        last_k_preds, last_k_scores = self.sess.run([self.model.k_preds, self.model.k_scores],
                                                    feed_dict={self.model.encoder_output: encoder_output,
                                                               self.model.decoder_input: preds})  # [batch_size * beam_size, beam_size]
        last_k_preds = last_k_preds.reshape([batch_size, -1])  # [batch_size, beam_size * beam_size]
        last_k_scores += scores[:, None]    # Add parents scores
        last_k_scores = last_k_scores.reshape([batch_size, -1])     # [batch_size, beam_size * beam_size]
        last_k_preds.t

    def evaluate(self):
        # Load data
        du = DataUtil(self.config)
        _, tmp = mkstemp()
        fd = codecs.open(tmp, 'w', 'utf8')
        for batch in du.get_test_batches():
            if config.test.beam_size == 1:
                Y = self.greedy_search(batch)
            else:
                Y = self.beam_search(batch)
            sents = du.indices_to_words(Y)
            for sent in sents:
                print(sent, file=fd)
        fd.close()

        # Remove BPE flag, if have.
        os.system("sed -r 's/(@@ )|(@@ ?$)//g' %s > %s" % (tmp, self.config.test.output_path))

        # Call a script to evaluate.
        os.system("perl multi-bleu.perl %s < %s" % (self.config.test.dst_path, self.config.test.output_path))

                                          
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', dest='config')
    args = parser.parse_args()
    # Read config
    config = AttrDict(yaml.load(open(args.config)))
    evaluator = Evaluator(config)
    evaluator.evaluate()
    print("Done")
