from torch import optim, nn
import torch
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import re

from ..utils import tolist
from ..distributions.distributions import Distribution


class Model(object):
    """
    This class is for training and testing a loss class.
    It requires a defined loss class, distributions to train, and optimizer for initialization.
    """

    def __init__(self, loss, test_loss=None, distributions=[],
                 optimizer=optim.Adam, optimizer_params={},
                 clip_grad_norm=None, clip_grad_value=None):
        """
        Args:
            loss (pixyz.losses.Loss):       Loss class for training.
            test_loss (pixyz.losses.Loss):  Loss class for testing.
            distributions (list[pixyz.distributions.Distribution]): List of class
            optimizer (torch.optim.Adam):   Optimization algorithm.
            optimizer_params (dict):        Parameters of optimizer
            clip_grad_norm (float):         Maximum allowed norm of the gradients.
            clip_grad_value (float):        Maximum allowed value of the gradients.
        """
        # set losses
        self.loss_cls = loss
        self.test_loss_cls = test_loss if (test_loss is not None) else loss

        # set distributions (for training)
        self.distributions = nn.ModuleList(tolist(distributions))

        # set params and optim
        params = self.distributions.parameters()
        self.optimizer = optimizer(params, **optimizer_params)

        self.clip_norm = clip_grad_norm
        self.clip_value = clip_grad_value

    def train(self, x_dict={}, **kwargs):
        """
        Args:
            x_dict (dict): Input data.
            **kwargs:

        Returns:
            torch.Tensor: Train loss value/

        """
        self.distributions.train(True)

        self.optimizer.zero_grad()
        loss = self.loss_cls.eval(x_dict, **kwargs)

        # calc backpropagation
        loss.backward()

        if self.clip_norm is not None:
            clip_grad_norm_(self.distributions.parameters(), self.clip_norm)
        if self.clip_value is not None:
            clip_grad_value_(self.distributions.parameters(), self.clip_value)

        # update params
        self.optimizer.step()

        return loss

    def test(self, x_dict={}, **kwargs):
        """
        Args:
            x_dict (dict): Input data.
            **kwargs:

        Returns:
            torch.Tensor: Test loss value.

        """
        self.distributions.train(False)

        with torch.no_grad():
            loss = self.test_loss_cls.eval(x_dict, **kwargs)

        return loss

    def __str__(self):
        prob_text = []
        func_text = []

        for prob in self.distributions._modules.values():
            if isinstance(prob, Distribution):
                prob_text.append(prob.prob_text)
            else:
                func_text.append(prob.__str__())

        text = "Distributions (for training): \n  {} \n".format(", ".join(prob_text))
        if len(func_text) > 0:
            text += "Deterministic functions (for training): \n  {} \n".format(", ".join(func_text))

        text += "Loss function: \n  {} \n".format(str(self.loss_cls))
        optimizer_text = re.sub('^', ' ' * 2, str(self.optimizer), flags=re.MULTILINE)
        text += "Optimizer: \n{}".format(optimizer_text)
        return text
