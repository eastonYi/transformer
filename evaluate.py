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
        pass

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

        # In case of BPE.
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
