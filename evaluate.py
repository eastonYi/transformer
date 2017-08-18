from __future__ import print_function
import codecs
import os
import tensorflow as tf
import numpy as np
import yaml
import time
import logging
from tempfile import mkstemp
from argparse import ArgumentParser

from model import Transformer
from utils import DataUtil, AttrDict


class Evaluator(object):
    """
    Evaluate the model.
    """
    def __init__(self, config):
        self.config = config

        # Load model
        # self.model = Model(config)
        self.model = Transformer(config)
        self.model.build_test_model()

        self.du = DataUtil(config)

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

    def beam_search(self, X):
        return self.sess.run(self.model.prediction, feed_dict={self.model.src_pl: X})

    def loss(self, X, Y):
        return self.sess.run(self.model.loss_sum, feed_dict={self.model.src_pl: X, self.model.dst_pl: Y})

    def translate(self):
        logging.info('Translate %s.' % self.config.test.src_path)
        _, tmp = mkstemp()
        fd = codecs.open(tmp, 'w', 'utf8')
        count = 0
        token_count = 0
        start = time.time()
        for X in self.du.get_test_batches():
            Y = self.beam_search(X)
            sents = self.du.indices_to_words(Y)
            for sent in sents:
                print(sent, file=fd)
            count += len(X)
            token_count += np.sum(np.not_equal(Y, 3))  # 3: </s>
            time_span = time.time() - start
            logging.info('{0} sentences ({1} tokens) processed in {2:.2f} minutes (speed: {3:.4f} sec/token).'.
                         format(count, token_count, time_span / 60, time_span / token_count))
        fd.close()
        # Remove BPE flag, if have.
        os.system("sed -r 's/(@@ )|(@@ ?$)//g' %s > %s" % (tmp, self.config.test.output_path))
        os.remove(tmp)
        logging.info('The result file was saved in %s.' % self.config.test.output_path)

    def ppl(self):
        if 'dst_path' not in self.config.test:
            logging.warning("Skip PPL calculation due to missing of parameter 'dst_path' in config file.")
            return
        logging.info('Calculate PPL for %s and %s.' % (self.config.test.src_path, self.config.test.dst_path))
        token_count = 0
        loss_sum = 0
        for batch in self.du.get_test_batches_with_target():
            X, Y = batch
            loss_sum += self.loss(X, Y)
            token_count += np.sum(np.greater(Y, 0))
        # Compute PPL
        logging.info('PPL: %.4f' % np.exp(loss_sum / token_count))

    def evaluate(self):
        self.translate()
        if 'cmd' in self.config.test:
            cmd = self.config.test.cmd
        else:
            cmd = 'perl multi-bleu.perl {ref} < {output}'
        # Call the script to evaluate.
        os.system(cmd.format(**{'ref': self.config.test.ori_dst_path, 'output': self.config.test.output_path}))
        self.ppl()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config')
    args = parser.parse_args()
    # Read config
    config = AttrDict(yaml.load(open(args.config)))
    # Logger
    logging.basicConfig(level=logging.INFO)
    evaluator = Evaluator(config)
    evaluator.evaluate()
    logging.info("Done")
