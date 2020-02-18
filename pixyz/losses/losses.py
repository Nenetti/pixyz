import abc
import sympy
import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

import numbers
from copy import deepcopy

from ..utils import tolist


class Loss(torch.nn.Module, metaclass=abc.ABCMeta):
    """Loss class. In Pixyz, all loss classes are required to inherit this class.

    Examples
    --------
    >>> import torch
    >>> from torch.nn import functional as F
    >>> from pixyz.distributions import Bernoulli, Normal
    >>> from pixyz.losses import StochasticReconstructionLoss, KullbackLeibler
    ...
    >>> # Set distributions
    >>> class Inference(Normal):
    ...     def __init__(self):
    ...         super().__init__(cond_var=["x"], var=["z"], name="q")
    ...         self.model_loc = torch.nn.Linear(128, 64)
    ...         self.model_scale = torch.nn.Linear(128, 64)
    ...     def forward(self, x, **kwargs):
    ...         return {"loc": self.model_loc(x), "scale": F.softplus(self.model_scale(x))}
    ...
    >>> class Generator(Bernoulli):
    ...     def __init__(self):
    ...         super().__init__(cond_var=["z"], var=["x"], name="p")
    ...         self.model = torch.nn.Linear(64, 128)
    ...     def forward(self, z, **kwargs):
    ...         return {"probs": torch.sigmoid(self.model(z))}
    ...
    >>> p = Generator()
    >>> q = Inference()
    >>> class NormalPrior(Normal):
    ...     def forward(self, **kwargs):
    ...         return {'loc': torch.zeros(1, 64), 'scale': torch.ones(1, 64)}
    >>> prior = NormalPrior(var=["z"], features_shape=[32], name="p_{prior}")
    ...
    >>> # Define a loss function (VAE)
    >>> reconst = StochasticReconstructionLoss(q, p)
    >>> kl = KullbackLeibler(q, prior)
    >>> loss_cls = (reconst - kl).mean()
    >>> print(loss_cls)
    mean \\left(- D_{KL} \\left[q(z|x)||p_{prior}(z) \\right] - \\mathbb{E}_{q(z|x)} \\left[\\log p(x|z) \\right] \\right)
    >>> # Evaluate this loss function
    >>> data = torch.randn(1, 128)  # Pseudo data
    >>> loss = loss_cls.eval({"x": data})
    >>> print(loss)  # doctest: +SKIP
    tensor(65.5939, grad_fn=<MeanBackward0>)

    """

    def __init__(self, input_var=None):
        """
        Parameters
        ----------
        input_var : :obj:`list` of :obj:`str`, defaults to None
            Input variables of this loss function.
            In general, users do not need to set them explicitly
            because these depend on the given distributions and each loss function.

        """
        super().__init__()
        self._input_var = input_var

    @property
    def input_var(self):
        """list: Input variables of this distribution."""
        return self._input_var

    @property
    @abc.abstractmethod
    def _symbol(self):
        raise NotImplementedError()

    @property
    def loss_text(self):
        return sympy.latex(self._symbol)

    def __str__(self):
        return self.loss_text

    def __repr__(self):
        return self.loss_text

    def __add__(self, other):
        return AddLoss(self, other)

    def __radd__(self, other):
        return AddLoss(other, self)

    def __sub__(self, other):
        return SubLoss(self, other)

    def __rsub__(self, other):
        return SubLoss(other, self)

    def __mul__(self, other):
        return MulLoss(self, other)

    def __rmul__(self, other):
        return MulLoss(other, self)

    def __truediv__(self, other):
        return DivLoss(self, other)

    def __rtruediv__(self, other):
        return DivLoss(other, self)

    def __neg__(self):
        return NegLoss(self)

    def abs(self):
        """Return an instance of :class:`pixyz.losses.losses.AbsLoss`.

        Returns
        -------
        pixyz.losses.losses.AbsLoss
            An instance of :class:`pixyz.losses.losses.AbsLoss`

        """
        return AbsLoss(self)

    def mean(self):
        """Return an instance of :class:`pixyz.losses.losses.BatchMean`.

        Returns
        -------
        pixyz.losses.losses.BatchMean
            An instance of :class:`pixyz.losses.BatchMean`

        """
        return BatchMean(self)

    def sum(self):
        """Return an instance of :class:`pixyz.losses.losses.BatchSum`.

        Returns
        -------
        pixyz.losses.losses.BatchSum
            An instance of :class:`pixyz.losses.losses.BatchSum`

        """
        return BatchSum(self)

    def detach(self):
        """Return an instance of :class:`pixyz.losses.losses.Detach`.

        Returns
        -------
        pixyz.losses.losses.Detach
            An instance of :class:`pixyz.losses.losses.Detach`

        """
        return Detach(self)

    def expectation(self, p, input_var=None, sample_shape=torch.Size()):
        """Return an instance of :class:`pixyz.losses.Expectation`.

        Parameters
        ----------
        p : pixyz.distributions.Distribution
            Distribution for sampling.

        input_var : list
            Input variables of this loss.

        sample_shape : :obj:`list` or :obj:`NoneType`, defaults to torch.Size()
            Shape of generating samples.

        Returns
        -------
        pixyz.losses.Expectation
            An instance of :class:`pixyz.losses.Expectation`

        """
        return Expectation(p, self, input_var=input_var, sample_shape=sample_shape)

    def eval(self, x_dict={}, return_dict=False, **kwargs):
        """Evaluate the value of the loss function given inputs (:attr:`x_dict`).

        Parameters
        ----------
        x_dict : :obj:`dict`, defaults to {}
            Input variables.
        return_dict : bool, default to False.
            Whether to return samples along with the evaluated value of the loss function.

        Returns
        -------
        loss : torch.Tensor
            the evaluated value of the loss function.
        x_dict : :obj:`dict`
            All samples generated when evaluating the loss function.
            If :attr:`return_dict` is False, it is not returned.

        """

        if not(set(list(x_dict.keys())) >= set(self._input_var)):
            raise ValueError("Input keys are not valid, expected {} but got {}.".format(self._input_var,
                                                                                        list(x_dict.keys())))

        loss, x_dict = self.forward(x_dict, **kwargs)

        if return_dict:
            return loss, x_dict

        return loss

    @abc.abstractmethod
    def forward(self, x_dict, **kwargs):
        """
        Parameters
        ----------
        x_dict : dict
            Input variables.

        Returns
        -------
        a tuple of :class:`pixyz.losses.Loss` and dict
        deterministically calcurated loss and updated all samples.
        """
        raise NotImplementedError()


class Divergence(Loss, abc.ABC):
    def __init__(self, p, q=None, input_var=None):
        """
        Parameters
        ----------
        p : pixyz.distributions.Distribution
            Distribution.
        q : pixyz.distributions.Distribution, defaults to None
            Distribution.
        input_var : :obj:`list` of :obj:`str`, defaults to None
            Input variables of this loss function.
            In general, users do not need to set them explicitly
            because these depend on the given distributions and each loss function.

        """
        if input_var is not None:
            _input_var = input_var
        else:
            _input_var = deepcopy(p.input_var)
            if q is not None:
                _input_var += deepcopy(q.input_var)
                _input_var = sorted(set(_input_var), key=_input_var.index)
        super().__init__(_input_var)
        self.p = p
        self.q = q


class ValueLoss(Loss):
    """
    This class contains a scalar as a loss value.

    If multiplying a scalar by an arbitrary loss class, this scalar is converted to the :class:`ValueLoss`.


    Examples
    --------
    >>> loss_cls = ValueLoss(2)
    >>> print(loss_cls)
    2
    >>> loss = loss_cls.eval()
    >>> print(loss)
    tensor(2.)

    """
    def __init__(self, loss1):
        super().__init__()
        self.original_value = loss1
        self.register_buffer('value', torch.tensor(loss1, dtype=torch.float))
        self._input_var = []

    def forward(self, x_dict={}, **kwargs):
        return self.value, x_dict

    @property
    def _symbol(self):
        return self.original_value


class Parameter(Loss):
    """
    This class defines a single variable as a loss class.

    It can be used such as a coefficient parameter of a loss class.

    Examples
    --------
    >>> loss_cls = Parameter("x")
    >>> print(loss_cls)
    x
    >>> loss = loss_cls.eval({"x": 2})
    >>> print(loss)
    2

    """
    def __init__(self, input_var):
        if not isinstance(input_var, str):
            raise ValueError()
        super().__init__(tolist(input_var))

    def forward(self, x_dict={}, **kwargs):
        return x_dict[self._input_var[0]], x_dict

    @property
    def _symbol(self):
        return sympy.Symbol(self._input_var[0])


class LossOperator(Loss):
    def __init__(self, loss1, loss2):
        super().__init__()
        _input_var = []

        if isinstance(loss1, Loss):
            _input_var += deepcopy(loss1.input_var)
        elif isinstance(loss1, numbers.Number):
            loss1 = ValueLoss(loss1)
        elif isinstance(loss2, type(None)):
            pass
        else:
            raise ValueError("{} cannot be operated with {}.".format(type(loss1), type(loss2)))

        if isinstance(loss2, Loss):
            _input_var += deepcopy(loss2.input_var)
        elif isinstance(loss2, numbers.Number):
            loss2 = ValueLoss(loss2)
        elif isinstance(loss2, type(None)):
            pass
        else:
            raise ValueError("{} cannot be operated with {}.".format(type(loss2), type(loss1)))

        _input_var = sorted(set(_input_var), key=_input_var.index)

        self._input_var = _input_var
        self.loss1 = loss1
        self.loss2 = loss2

    def forward(self, x_dict={}, **kwargs):
        if not isinstance(self.loss1, type(None)):
            loss1, x1 = self.loss1.forward(x_dict, **kwargs)
        else:
            loss1 = 0
            x1 = {}

        if not isinstance(self.loss2, type(None)):
            loss2, x2 = self.loss2.forward(x_dict, **kwargs)
        else:
            loss2 = 0
            x2 = {}

        x1.update(x2)

        return loss1, loss2, x1


class AddLoss(LossOperator):
    """
    Apply the `add` operation to the two losses.

    Examples
    --------
    >>> loss_cls_1 = ValueLoss(2)
    >>> loss_cls_2 = Parameter("x")
    >>> loss_cls = loss_cls_1 + loss_cls_2  # equals to AddLoss(loss_cls_1, loss_cls_2)
    >>> print(loss_cls)
    x + 2
    >>> loss = loss_cls.eval({"x": 3})
    >>> print(loss)
    tensor(5.)

    """
    @property
    def _symbol(self):
        return self.loss1._symbol + self.loss2._symbol

    def forward(self, x_dict={}, **kwargs):
        loss1, loss2, x_dict = super().forward(x_dict, **kwargs)
        return loss1 + loss2, x_dict


class SubLoss(LossOperator):
    """
    Apply the `sub` operation to the two losses.

    Examples
    --------
    >>> loss_cls_1 = ValueLoss(2)
    >>> loss_cls_2 = Parameter("x")
    >>> loss_cls = loss_cls_1 - loss_cls_2  # equals to SubLoss(loss_cls_1, loss_cls_2)
    >>> print(loss_cls)
    2 - x
    >>> loss = loss_cls.eval({"x": 4})
    >>> print(loss)
    tensor(-2.)
    >>> loss_cls = loss_cls_2 - loss_cls_1  # equals to SubLoss(loss_cls_2, loss_cls_1)
    >>> print(loss_cls)
    x - 2
    >>> loss = loss_cls.eval({"x": 4})
    >>> print(loss)
    tensor(2.)

    """
    @property
    def _symbol(self):
        return self.loss1._symbol - self.loss2._symbol

    def forward(self, x_dict={}, **kwargs):
        loss1, loss2, x_dict = super().forward(x_dict, **kwargs)
        return loss1 - loss2, x_dict


class MulLoss(LossOperator):
    """
    Apply the `mul` operation to the two losses.

    Examples
    --------
    >>> loss_cls_1 = ValueLoss(2)
    >>> loss_cls_2 = Parameter("x")
    >>> loss_cls = loss_cls_1 * loss_cls_2  # equals to MulLoss(loss_cls_1, loss_cls_2)
    >>> print(loss_cls)
    2 x
    >>> loss = loss_cls.eval({"x": 4})
    >>> print(loss)
    tensor(8.)

    """
    @property
    def _symbol(self):
        return self.loss1._symbol * self.loss2._symbol

    def forward(self, x_dict={}, **kwargs):
        loss1, loss2, x_dict = super().forward(x_dict, **kwargs)
        return loss1 * loss2, x_dict


class DivLoss(LossOperator):
    """
    Apply the `div` operation to the two losses.

    Examples
    --------
    >>> loss_cls_1 = ValueLoss(2)
    >>> loss_cls_2 = Parameter("x")
    >>> loss_cls = loss_cls_1 / loss_cls_2  # equals to DivLoss(loss_cls_1, loss_cls_2)
    >>> print(loss_cls)
    \\frac{2}{x}
    >>> loss = loss_cls.eval({"x": 4})
    >>> print(loss)
    tensor(0.5000)
    >>> loss_cls = loss_cls_2 / loss_cls_1  # equals to DivLoss(loss_cls_2, loss_cls_1)
    >>> print(loss_cls)
    \\frac{x}{2}
    >>> loss = loss_cls.eval({"x": 4})
    >>> print(loss)
    tensor(2.)


    """
    @property
    def _symbol(self):
        return self.loss1._symbol / self.loss2._symbol

    def forward(self, x_dict={}, **kwargs):
        loss1, loss2, x_dict = super().forward(x_dict, **kwargs)
        return loss1 / loss2, x_dict


class MinLoss(LossOperator):
    r"""
    Apply the `min` operation to the loss.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> from pixyz.losses.losses import ValueLoss, Parameter, MinLoss
    >>> loss_min= MinLoss(ValueLoss(3), ValueLoss(1))
    >>> print(loss_min)
    min \left(3, 1\right)
    >>> print(loss_min.eval())
    tensor(1.)
    """
    def __init__(self, loss1, loss2):
        super().__init__(loss1, loss2)

    @property
    def _symbol(self):
        return sympy.Symbol(f"min \\left({self.loss1.loss_text}, {self.loss2.loss_text}\\right)")

    def forward(self, x_dict={}, **kwargs):
        loss1, loss2, x_dict = super().forward(x_dict, **kwargs)
        return torch.min(loss1, loss2), x_dict


class MaxLoss(LossOperator):
    r"""
    Apply the `max` operation to the loss.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> from pixyz.losses.losses import ValueLoss, MaxLoss
    >>> loss_max= MaxLoss(ValueLoss(3), ValueLoss(1))
    >>> print(loss_max)
    max \left(3, 1\right)
    >>> print(loss_max.eval())
    tensor(3.)
    """
    def __init__(self, loss1, loss2):
        super().__init__(loss1, loss2)

    @property
    def _symbol(self):
        return sympy.Symbol(f"max \\left({self.loss1.loss_text}, {self.loss2.loss_text}\\right)")

    def forward(self, x_dict={}, **kwargs):
        loss1, loss2, x_dict = super().forward(x_dict, **kwargs)
        return torch.max(loss1, loss2), x_dict


class LossSelfOperator(Loss):
    def __init__(self, loss1):
        super().__init__()
        _input_var = []

        if isinstance(loss1, type(None)):
            raise ValueError()

        if isinstance(loss1, Loss):
            _input_var = deepcopy(loss1.input_var)
        elif isinstance(loss1, numbers.Number):
            loss1 = ValueLoss(loss1)
        else:
            raise ValueError()

        self._input_var = _input_var
        self.loss1 = loss1

    def loss_train(self, x_dict={}, **kwargs):
        return self.loss1.loss_train(x_dict, **kwargs)

    def loss_test(self, x_dict={}, **kwargs):
        return self.loss1.loss_test(x_dict, **kwargs)


class NegLoss(LossSelfOperator):
    """
    Apply the `neg` operation to the loss.

    Examples
    --------
    >>> loss_cls_1 = Parameter("x")
    >>> loss_cls = -loss_cls_1  # equals to NegLoss(loss_cls_1)
    >>> print(loss_cls)
    - x
    >>> loss = loss_cls.eval({"x": 4})
    >>> print(loss)
    -4

    """
    @property
    def _symbol(self):
        return -self.loss1._symbol

    def forward(self, x_dict={}, **kwargs):
        loss, x_dict = self.loss1.forward(x_dict, **kwargs)
        return -loss, x_dict


class AbsLoss(LossSelfOperator):
    """
    Apply the `abs` operation to the loss.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> from pixyz.losses import LogProb
    >>> class NormalP(Normal):
    ...     def forward(self, **kwargs):
    ...         return {'loc': torch.zeros(1, 10), 'scale': torch.ones(1, 10)}
    >>> p = NormalP(var=["x"], features_shape=[10])
    >>> loss_cls = LogProb(p).abs() # equals to AbsLoss(LogProb(p))
    >>> print(loss_cls)
    |\\log p(x)|
    >>> sample_x = torch.randn(2, 10) # Psuedo data
    >>> loss = loss_cls.eval({"x": sample_x})
    >>> print(loss) # doctest: +SKIP
    tensor([12.9894, 15.5280])

    """
    @property
    def _symbol(self):
        return sympy.Symbol("|{}|".format(self.loss1.loss_text))

    def forward(self, x_dict={}, **kwargs):
        loss, x_dict = self.loss1.forward(x_dict, **kwargs)
        return loss.abs(), x_dict


class BatchMean(LossSelfOperator):
    r"""
    Average a loss class over given batch data.

    .. math::

        \mathbb{E}_{p_{data}(x)}[\mathcal{L}(x)] \approx \frac{1}{N}\sum_{i=1}^N \mathcal{L}(x_i),

    where :math:`x_i \sim p_{data}(x)` and :math:`\mathcal{L}` is a loss function.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> from pixyz.losses import LogProb
    >>> class NormalP(Normal):
    ...     def forward(self, **kwargs):
    ...         return {'loc': torch.zeros(1, 10), 'scale': torch.ones(1, 10)}
    >>> p = NormalP(var=["x"], features_shape=[10])
    >>> loss_cls = LogProb(p).mean() # equals to BatchMean(LogProb(p))
    >>> print(loss_cls)
    mean \left(\log p(x) \right)
    >>> sample_x = torch.randn(2, 10) # Psuedo data
    >>> loss = loss_cls.eval({"x": sample_x})
    >>> print(loss) # doctest: +SKIP
    tensor(-14.5038)
    """

    @property
    def _symbol(self):
        return sympy.Symbol("mean \\left({} \\right)".format(self.loss1.loss_text))  # TODO: fix it

    def forward(self, x_dict={}, **kwargs):
        loss, x_dict = self.loss1.forward(x_dict, **kwargs)
        return loss.mean(), x_dict


class BatchSum(LossSelfOperator):
    r"""
    Summation a loss class over given batch data.

    .. math::

        \sum_{i=1}^N \mathcal{L}(x_i),

    where :math:`x_i \sim p_{data}(x)` and :math:`\mathcal{L}` is a loss function.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> from pixyz.losses import LogProb
    >>> class NormalP(Normal):
    ...     def forward(self, **kwargs):
    ...         return {'loc': torch.zeros(1, 10), 'scale': torch.ones(1, 10)}
    >>> p = NormalP(var=["x"], features_shape=[10])
    >>> loss_cls = LogProb(p).sum() # equals to BatchSum(LogProb(p))
    >>> print(loss_cls)
    sum \left(\log p(x) \right)
    >>> sample_x = torch.randn(2, 10) # Psuedo data
    >>> loss = loss_cls.eval({"x": sample_x})
    >>> print(loss) # doctest: +SKIP
    tensor(-31.9434)
    """

    @property
    def _symbol(self):
        return sympy.Symbol("sum \\left({} \\right)".format(self.loss1.loss_text))  # TODO: fix it

    def forward(self, x_dict={}, **kwargs):
        loss, x_dict = self.loss1.forward(x_dict, **kwargs)
        return loss.sum(), x_dict


class Detach(LossSelfOperator):
    r"""
    Apply the `detach` method to the loss.

    """

    @property
    def _symbol(self):
        return sympy.Symbol("detach \\left({} \\right)".format(self.loss1.loss_text))  # TODO: fix it?

    def forward(self, x_dict={}, **kwargs):
        loss, x_dict = self.loss1.forward(x_dict, **kwargs)
        return loss.detach(), x_dict


class Expectation(Loss):
    r"""
    Expectation of a given function (Monte Carlo approximation).

    .. math::

        \mathbb{E}_{p(x)}[f(x)] \approx \frac{1}{L}\sum_{l=1}^L f(x_l),
         \quad \text{where}\quad x_l \sim p(x).

    Note that :math:`f` doesn't need to be able to sample, which is known as the law of the unconscious statistician
    (LOTUS).

    Therefore, in this class, :math:`f` is assumed to :attr:`pixyz.Loss`.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal, Bernoulli
    >>> from pixyz.losses import LogProb
    >>> class NormalQ(Normal):
    ...     def forward(self, x, **kwargs):
    ...         return {'loc': x, 'scale': torch.ones(1, 10)}
    >>> q = NormalQ(var=["z"], cond_var=["x"], features_shape=[10])
    >>> class NormalP(Normal):
    ...     def forward(self, z, **kwargs):
    ...         return {'loc': z, 'scale': torch.ones(1, 10)}
    >>> p = NormalP(var=["x"], cond_var=["z"], features_shape=[10])
    >>> loss_cls = LogProb(p).expectation(q) # equals to Expectation(q, LogProb(p))
    >>> print(loss_cls)
    \mathbb{E}_{p(z|x)} \left[\log p(x|z) \right]
    >>> sample_x = torch.randn(2, 10) # Psuedo data
    >>> loss = loss_cls.eval({"x": sample_x})
    >>> print(loss) # doctest: +SKIP
    tensor([-12.8181, -12.6062])
    >>> loss_cls = LogProb(p).expectation(q, sample_shape=(5,))
    >>> loss = loss_cls.eval({"x": sample_x})
    >>> print(loss) # doctest: +SKIP
    >>> class BernoulliQ(Bernoulli):
    ...     def forward(self, **kwargs):
    ...         return {'probs': torch.ones(1, 10)*0.5}
    >>> q = BernoulliQ(var=["x"], features_shape=[10])
    >>> class BernoulliP(Bernoulli):
    ...     def forward(self, **kwargs):
    ...         return {'probs': torch.ones(1, 10)*0.3}
    >>> p = BernoulliQ(var=["x"], features_shape=[10])
    >>> loss_cls = p.log_prob().expectation(q, sample_shape=[64])
    >>> train_loss = loss_cls.eval()
    >>> print(train_loss) # doctest: +SKIP
    tensor([46.7559])
    >>> eval_loss = loss_cls.eval(test_mode=True)
    >>> print(eval_loss) # doctest: +SKIP
    tensor([-7.6047])

    """

    def __init__(self, p, f, input_var=None, sample_shape=torch.Size([1]), reparam=True):

        if input_var is None:
            input_var = list(set(p.input_var) | set(f.input_var) - set(p.var))
        super().__init__(input_var=input_var)
        self.p = p
        self.f = f
        self.sample_shape = torch.Size(sample_shape)
        self.reparam = reparam

    @property
    def _symbol(self):
        p_text = "{" + self.p.prob_text + "}"
        return sympy.Symbol("\\mathbb{{E}}_{} \\left[{} \\right]".format(p_text, self.f.loss_text))

    def forward(self, x_dict={}, **kwargs):
        samples_dicts = [self.p.sample(x_dict, reparam=self.reparam, return_all=True) for i in range(self.sample_shape.numel())]

        loss_and_dicts = [self.f.eval(samples_dict, return_dict=True, **kwargs) for
                          samples_dict in samples_dicts]

        losses = [loss for loss, loss_sample_dict in loss_and_dicts]
        # sum over sample_shape
        loss = torch.stack(losses).mean(dim=0)
        samples_dicts[0].update(loss_and_dicts[0][1])

        return loss, samples_dicts[0]


def REINFORCE(p, f, b=ValueLoss(0), input_var=None, sample_shape=torch.Size([1]), reparam=True):
    r"""
    Surrogate Loss for Policy Gradient Method (REINFORCE) with a given reward function :math:`f` and a given baseline :math:`b`.

    .. math::

        \mathbb{E}_{p(x)}[detach(f(x)-b(x))\log p(x)+f(x)-b(x)].

    in this function, :math:`f` and :math:`b` is assumed to :attr:`pixyz.Loss`.

    Parameters
    ----------
    p : :class:`pixyz.distributions.Distribution`
            Distribution for expectation.
    f : :class:`pixyz.losses.Loss`
            reward function
    b : :class:`pixyz.losses.Loss`
            baseline function

    Returns
    -------
    surrogate_loss : :class:`pixyz.losses.Loss`
            policy gradient can be calcurated from a gradient of this surrogate loss.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal, Bernoulli
    >>> from pixyz.losses import LogProb
    >>> class BernoulliQ(Bernoulli):
    ...     def forward(self, **kwargs):
    ...         return {'probs': torch.ones(1, 10)*0.5}
    >>> q = BernoulliQ(var=["x"], features_shape=[10])
    >>> class BernoulliP(Bernoulli):
    ...     def forward(self, **kwargs):
    ...         return {'probs': torch.ones(1, 10)*0.3}
    >>> p = BernoulliQ(var=["x"], features_shape=[10])
    >>> loss_cls = REINFORCE(q, p.log_prob(), sample_shape=[64])
    >>> train_loss = loss_cls.eval(test_mode=True)
    >>> print(train_loss) # doctest: +SKIP
    tensor([46.7559])
    >>> loss_cls = p.log_prob().expectation(q, sample_shape=[64])
    >>> test_loss = loss_cls.eval()
    >>> print(test_loss) # doctest: +SKIP
    tensor([-7.6047])

    """
    return Expectation(p, (f - b).detach() * p.log_prob() + (f - b), input_var, sample_shape, reparam=reparam)


class DataParalleledLoss(Loss):
    r"""
    Loss class wrapper of torch.nn.DataParallel. It can be used as the original loss class.
    `eval` & `forward` methods support data-parallel running.

    Examples
    --------
    >>> import torch
    >>> from torch import optim
    >>> from torch.nn import functional as F
    >>> from pixyz.distributions import Bernoulli, Normal
    >>> from pixyz.losses import StochasticReconstructionLoss, KullbackLeibler, DataParalleledLoss
    >>> from pixyz.models import Model
    >>> used_gpu_i = set()
    >>> used_gpu_g = set()
    >>> # Set distributions (Distribution API)
    >>> class Inference(Normal):
    ...     def __init__(self):
    ...         super().__init__(cond_var=["x"], var=["z"], name="q")
    ...         self.model_loc = torch.nn.Linear(128, 64)
    ...         self.model_scale = torch.nn.Linear(128, 64)
    ...     def forward(self, x, **kwargs):
    ...         used_gpu_i.add(x.device.index)
    ...         return {"loc": self.model_loc(x), "scale": F.softplus(self.model_scale(x))}
    >>> class Generator(Bernoulli):
    ...     def __init__(self):
    ...         super().__init__(cond_var=["z"], var=["x"], name="p")
    ...         self.model = torch.nn.Linear(64, 128)
    ...     def forward(self, z, **kwargs):
    ...         used_gpu_g.add(z.device.index)
    ...         return {"probs": torch.sigmoid(self.model(z))}
    >>> p = Generator()
    >>> q = Inference()
    >>> class NormalPrior(Normal):
    ...     def forward(self, **kwargs):
    ...         return {'loc': torch.zeros(1, 64), 'scale': torch.ones(1, 64)}
    >>> prior = NormalPrior(var=["z"], features_shape=[64], name="p_{prior}")
    >>> # Define a loss function (Loss API)
    >>> reconst = StochasticReconstructionLoss(q, p)
    >>> kl = KullbackLeibler(q, prior)
    >>> batch_loss_cls = (reconst - kl)
    >>> # device settings
    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> device_count = torch.cuda.device_count()
    >>> if device_count > 1:
    ...     loss_cls = DataParalleledLoss(batch_loss_cls).mean().to(device)
    ... else:
    ...     loss_cls = batch_loss_cls.mean().to(device)
    >>> # Set a model (Model API)
    >>> model = Model(loss=loss_cls, distributions=[p, q],
    ...               optimizer=optim.Adam, optimizer_params={"lr": 1e-3})
    >>> # Train and test the model
    >>> data = torch.randn(2, 128).to(device)  # Pseudo data
    >>> train_loss = model.train({"x": data})
    >>> expected = set(range(device_count)) if torch.cuda.is_available() else {None}
    >>> assert used_gpu_i==expected
    >>> assert used_gpu_g==expected
    """
    def __init__(self, loss, distributed=False, **kwargs):
        super().__init__(loss.input_var)
        if distributed:
            self.paralleled = DistributedDataParallel(loss, **kwargs)
        else:
            self.paralleled = DataParallel(loss, **kwargs)

    def forward(self, x_dict, **kwargs):
        return self.paralleled.forward(x_dict, **kwargs)

    @property
    def _symbol(self):
        return self.paralleled.module._symbol

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.paralleled.module, name)
