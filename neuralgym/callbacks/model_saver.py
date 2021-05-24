"""model_saver"""
import os

from . import PeriodicCallback, CallbackLoc
from ..utils.logger import callback_log


class ModelSaver(PeriodicCallback):
    """Save model to file at every pstep step_start.

    Args:
        pstep (int): Save to model every pstep.
        saver: Tensorflow saver.
        dump_prefix (str): Prefix for saving model files.

    """

    def __init__(self, pstep, saver, dump_prefix, train_spe=None, save_every_epochs=50, op_lr=False, optim=None):
        super().__init__(CallbackLoc.step_start, pstep)
        self._saver = saver
        self._dump_prefix = dump_prefix ; self.train_spe = train_spe ; self.see = save_every_epochs
        # self.optim = optim ; self.op_lr = op_lr
        # self.best_losses = {} 
        # self.best_losses['d_loss'] = 999.; self.best_losses['g_loss'] = 999.; self.best_losses['avg_loss'] = 999.
        dump_dir = os.path.dirname(self._dump_prefix)
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
            callback_log('Initialize ModelSaver: mkdirs %s.' % dump_dir)
        '''
        # make two folders to save best D, G, and avg loss
        self.dump_dir_d = os.path.join(os.path.join(dump_dir, 'best_D'))
        if not os.path.exists(self.dump_dir_d):
            os.makedirs(self.dump_dir_d)

        self.dump_dir_g = os.path.join(os.path.join(dump_dir, 'best_G'))
        if not os.path.exists(self.dump_dir_g):
            os.makedirs(self.dump_dir_g)
        
        self.dump_dir_avg = os.path.join(os.path.join(dump_dir, 'best_avg'))
        if not os.path.exists(self.dump_dir_avg):
            os.makedirs(self.dump_dir_avg)
        '''

    def run(self, sess, step):
        '''
        if self.op_lr:
            g_lr = sess.run(self.optim['g']._lr)
            d_lr = sess.run(self.optim['d']._lr)
            callback_log('At step {}, lr: g: {}, d: {}.'.format(
                step, g_lr, d_lr))
        '''
        # save the best loss 

        # save model
        if step != 0 and int(step/self.train_spe)%self.see == 0:
            callback_log('Trigger ModelSaver: Save model to {}-{}.'.format(
                self._dump_prefix, step))
            self._saver.save(sess, self._dump_prefix, global_step=step)
