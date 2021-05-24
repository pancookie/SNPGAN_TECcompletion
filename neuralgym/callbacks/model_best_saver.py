"""model_best_saver"""
import os

from . import PeriodicCallback, CallbackLoc
from ..utils.logger import callback_log


class ModelBestSaver(PeriodicCallback):
    """Save best model to file at every pstep step_end.

    Args:
        pstep (int): Save to model every pstep.
        saver: Tensorflow saver.
        dump_prefix (str): Prefix for saving model files.

    """

    def __init__(self, pstep, saver, dump_prefix):
        super().__init__(CallbackLoc.step_end, pstep)
        self._saver = saver
        self._dump_prefix = dump_prefix
        dump_dir = os.path.dirname(self._dump_prefix)
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
            callback_log('Initialize ModelSaver: mkdirs %s.' % dump_dir)
        # make two folders to save best G
        self.dump_dir_g = os.path.join(os.path.join(dump_dir, 'best_G'))
        if not os.path.exists(self.dump_dir_g):
            os.makedirs(self.dump_dir_g)
            callback_log('Initialize ModelBestSaver: mkdirs %s.' % self.dump_dir_g)

    def run(self, sess, step):
        # save the best loss 
        if step != 0:
            # remove the previous saved model
            os.system('rm {}'.format(os.path.join(self.dump_dir_g, 'snap-*')))
            callback_log('Trigger ModelBestSaver: Save model to {}-{}.'.format(
                os.path.join(self.dump_dir_g, 'snap'), step))
            self._saver.save(sess, os.path.join(self.dump_dir_g, 'snap'), global_step=step)
