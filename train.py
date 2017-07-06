from __future__ import print_function
import yaml
import time
from argparse import ArgumentParser
import tensorflow as tf

from utils import DataUtil, AttrDict
from model import Model


def train(config):
    """Train a model with a config file."""
    du = DataUtil(config=config)
    model = Model(config=config)
    model.build_train_model()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    with model.graph.as_default():
        saver = tf.train.Saver(var_list=tf.global_variables())
        summary_writer = tf.summary.FileWriter(config.train.logdir, graph=model.graph)

        # saver_partial = tf.train.Saver(var_list=[v for v in tf.trainable_variables() if 'Adam' not in v.name])

        with tf.Session(config=sess_config) as sess:
            # Initialize all variables.
            sess.run(tf.global_variables_initializer())
            # saver_partial.restore(sess, tf.train.latest_checkpoint(config.train.logdir))
            # print('Restore partial model from %s.' % config.train.logdir)
            try:
                saver.restore(sess, tf.train.latest_checkpoint(config.train.logdir))
                print('Restore model from %s.' % config.train.logdir)
            except:
                print('Failed to reload model.')
            for epoch in range(1, config.train.num_epochs+1):
                for batch in du.get_training_batches_with_buckets():
                    start_time = time.time()
                    step, lr, gnorm, loss, acc, summary, _ = sess.run([model.global_step, model.learning_rate, model.grads_norm,
                                                                       model.loss, model.acc, model.summary_op, model.train_op],
                                                                      feed_dict={model.src_pl: batch[0], model.dst_pl: batch[1]})
                    print('epoch: {0}\tstep: {1}\tlr: {2:.6f}\tgnorm: {3:.4f}\tloss: {4:.4f}\tacc: {5:.4f}\ttime: {6:.4f}'.
                          format(epoch, step, lr, gnorm, loss, acc, time.time()-start_time))
                    summary_writer.add_summary(summary, global_step=step)
                    # Save model
                    if step % config.train.save_freq == 0:
                        mp = config.train.logdir + '/model_epoch_%d_step_%d' % (epoch, step)
                        saver.save(sess, mp)
                        print('Save model in %s.' % mp)
            print("Done")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', dest='config')
    args = parser.parse_args()
    # Read config
    config = AttrDict(yaml.load(open(args.config)))
    # Train
    train(config)
