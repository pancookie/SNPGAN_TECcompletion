"""model_lr"""

from . import PeriodicCallback, CallbackLoc
from ..utils.logger import callback_log


class LRDisplayer(PeriodicCallback):
    """ display learning rate value

    Args:
        pstep (int): Save to model every pstep.
        optimizer: 
    """

    def __init__(self, pstep, optimizer):
        super().__init__(CallbackLoc.step_start, pstep)
        self.optim = optimizer

    def run(self, sess, step):
        callback_log('At step {}, lr: {}.'.format(
            step, sess.run(self.optim._lr)))
