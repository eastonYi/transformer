"""
Written by Chunqi Wang in July 2017.
"""
from __future__ import print_function
import codecs
import os
import tensorflow as tf
import numpy as np
import yaml
import time
import logging
import subprocess
from tempfile import mkstemp
from argparse import ArgumentParser

from model import Transformer
from utils import DataReader, AttrDict, expand_feed_dict


class Evaluator(object):
    """
    Evaluate the model.
    """
    def __init__(self):
        pass

    def init_from_config(self, config):
        # self.model = Model(config)
        self.model = Transformer(config, config.test.devices)
        self.model.build_test_model()

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.allow_soft_placement = True
        self.sess = tf.Session(config=sess_config, graph=self.model.graph)
        # Restore model.
        self.model.saver.restore(self.sess, tf.train.latest_checkpoint(config.train.logdir))

        self.data_reader = DataReader(config)

    def init_from_existed(self, model, sess, data_reader):
        assert model.graph == sess.graph
        self.sess = sess
        self.model = model
        self.data_reader = data_reader

    def beam_search(self, X):
        return self.sess.run(self.model.prediction, feed_dict=expand_feed_dict({self.model.src_pls: X}))

    def loss(self, X, Y):
        return self.sess.run(self.model.loss_sum, feed_dict=expand_feed_dict({self.model.src_pls: X, self.model.dst_pls: Y}))

    def translate(self, src_path, output_path, batch_size):
        logging.info('Translate %s.' % src_path)
        _, tmp = mkstemp()
        fd = codecs.open(tmp, 'w', 'utf8')
        count = 0
        token_count = 0
        start = time.time()
        for X in self.data_reader.get_test_batches(src_path, batch_size):
            Y = self.beam_search(X)
            sents = self.data_reader.indices_to_words(Y)
            assert len(X) == len(sents)
            for sent in sents:
                print(sent, file=fd)
            count += len(X)
            token_count += np.sum(np.not_equal(Y, 3))  # 3: </s>
            time_span = time.time() - start
            logging.info('{0} sentences ({1} tokens) processed in {2:.2f} minutes (speed: {3:.4f} sec/token).'.
                         format(count, token_count, time_span / 60, time_span / token_count))
        fd.close()
        # Remove BPE flag, if have.
        os.system("sed -r 's/(@@ )|(@@ ?$)//g' %s > %s" % (tmp, output_path))
        os.remove(tmp)
        logging.info('The result file was saved in %s.' % output_path)

    def ppl(self, src_path, dst_path, batch_size):
        logging.info('Calculate PPL for %s and %s.' % (src_path, dst_path))
        token_count = 0
        loss_sum = 0
        for batch in self.data_reader.get_test_batches_with_target(src_path, dst_path, batch_size):
            X, Y = batch
            loss_sum += self.loss(X, Y)
            token_count += np.sum(np.greater(Y, 0))
        # Compute PPL
        ppl = np.exp(loss_sum / token_count)
        logging.info('PPL: %.4f' % ppl)
        return ppl

    def evaluate(self, batch_size, **kargs):
        """Evaluate the model on dev set."""
        src_path = kargs['src_path']
        ref_path = kargs['ref_path']
        output_path = kargs['output_path']
        cmd = kargs['cmd'] if 'cmd' in kargs else\
            "perl multi-bleu.perl {ref} < {output} 2>/dev/null | awk '{{print($3)}}' | awk -F, '{{print $1}}'"
        self.translate(src_path, output_path, batch_size)
        bleu = subprocess.getoutput(cmd.format(**{'ref': ref_path, 'output': output_path}))
        logging.info('BLEU: {}'.format(bleu))
        if 'dst_path' in kargs:
            self.ppl(src_path, kargs['dst_path'], batch_size)

        return float(bleu)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config')
    args = parser.parse_args()
    # Read config
    config = AttrDict(yaml.load(open(args.config)))
    # Logger
    logging.basicConfig(level=logging.INFO)
    evaluator = Evaluator()
    evaluator.init_from_config(config)
    for attr in config.test:
        if attr.startswith('set'):
            evaluator.evaluate(config.test.batch_size, **config.test[attr])
    logging.info("Done")
