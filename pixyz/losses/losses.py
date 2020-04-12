import abc
import sympy
import torch

import numbers
from copy import deepcopy

from ..variables import Variables


class AbstractLoss(object, metaclass=abc.ABCMeta):

    def __init__(self, input_var=[]):
        self._input_var = input_var

    @property
    def input_var(self):
        """list: Input variables of this distribution."""
        return self._input_var

    def eval(self, variables, return_dict=False, **kwargs):
        """Evaluate the value of the loss function given inputs (:attr:`variables`).

        Parameters
        ----------
        variables : :obj:`dict`, defaults to {}
            Input variables.
        return_dict : bool, default to False.
            Whether to return samples along with the evaluated value of the loss function.

        Returns
        -------
        loss : torch.Tensor
            the evaluated value of the loss function.
        variables : :obj:`dict`
            All samples generated when evaluating the loss function.
            If :attr:`return_dict` is False, it is not returned.

        """
        if not (set(list(variables.keys())) >= set(self._input_var)):
            raise ValueError("Input keys are not valid, expected {} but got {}.".format(self._input_var,
                                                                                        list(variables.keys())))

        loss, variables = self._get_eval(variables, **kwargs)

        if return_dict:
            return loss, variables

        return loss

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

    @abc.abstractmethod
    def print_arithmetic(self, n=2):
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_eval(self, variables, **kwargs):
        """
        Args:
            variables (Variables):

        """
        raise NotImplementedError()

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


class Loss(AbstractLoss):
    """
    Loss class. In Pixyz, all loss classes are required to inherit this class.

    """

    def __init__(self, p, q=None, input_var=None):
        """
        Args:
            p (pixyz.distributions.Distribution): Distribution.
            q (pixyz.distributions.Distribution): Distribution.
            input_var (list or str): Input variables of this loss function.

        """
        self.p = p
        self.q = q
        self._input_var = input_var
        if self._input_var is None:
            input_var = deepcopy(p.input_var)
            if q is not None:
                input_var += deepcopy(q.input_var)
                input_var = sorted(set(input_var), key=input_var.index)
            self._input_var = input_var

        super(Loss, self).__init__(input_var)

    def print_arithmetic(self, n=2):
        return f"\n{' ' * n}{self.__class__.__name__}({self.input_var}\n)"


class ValueLoss(AbstractLoss):
    """
    This class contains a scalar as a loss value.
    If multiplying a scalar by an arbitrary loss class, this scalar is converted to the :class:`ValueLoss`.

    Examples:
        >>> loss_cls = ValueLoss(2)
        >>> print(loss_cls)
        2
        >>> loss = loss_cls.eval()
        >>> print(loss)
        2

    """

    def __init__(self, loss1):
        super(ValueLoss, self).__init__()
        self.loss1 = loss1

    def _get_eval(self, variables={}, **kwargs):
        return self.loss1, variables

    @property
    def _symbol(self):
        return self.loss1

    def print_arithmetic(self, n=2):
        return f"\n{' ' * n}{self.__class__.__name__}({self.loss1.print_arithmetic(n + 2)}\n{' ' * n})"


class Parameter(AbstractLoss):
    """
    This class defines a single variable as a loss class.
    It can be used such as a coefficient parameter of a loss class.

    Examples:
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
        super(Parameter, self).__init__(input_var)

    def _get_eval(self, variables={}, **kwargs):
        return variables[self._input_var], variables

    @property
    def _symbol(self):
        return sympy.Symbol(self._input_var[0])

    def print_arithmetic(self, n=2):
        return f"\n{' ' * n}{self.__class__.__name__}({self.input_var}\n)"


class SetLoss(AbstractLoss):
    def __init__(self, loss):
        super(SetLoss, self).__init__(loss.input_var)
        self.loss = loss

    def __getattr__(self, name):
        getattr(self.loss, name)

    def _get_eval(self, variables, **kwargs):
        return self.loss._get_eval(variables, **kwargs)

    @property
    def _symbol(self):
        return self.loss._symbol

    def print_arithmetic(self, n=2):
        return f"\n{' ' * n}{self.__class__.__name__}({self.loss.print_arithmetic(n + 2)}\n{' ' * n})"


class MultiLossOperator(AbstractLoss):
    """
    Apply "Four arithmetic operations" ("add", "sub", "mul" "div")

    """

    def __init__(self, loss1, loss2):
        input_var = []

        if isinstance(loss1, AbstractLoss):
            input_var += deepcopy(loss1.input_var)
        elif isinstance(loss1, numbers.Number):
            loss1 = ValueLoss(loss1)
        else:
            raise ValueError(f"{type(loss1)} cannot be operated with {type(loss2)}.")

        if isinstance(loss2, AbstractLoss):
            input_var += deepcopy(loss2.input_var)
        elif isinstance(loss2, numbers.Number):
            loss2 = ValueLoss(loss2)
        else:
            raise ValueError(f"{type(loss2)} cannot be operated with {type(loss1)}.")

        input_var = sorted(set(input_var), key=input_var.index)

        super(MultiLossOperator, self).__init__(input_var)
        self.loss1 = loss1
        self.loss2 = loss2

    def _get_eval(self, variables, **kwargs):
        loss1, x1 = self.loss1._get_eval(variables, **kwargs)
        loss2, x2 = self.loss2._get_eval(variables, **kwargs)
        x1.update(x2)

        return loss1, loss2, x1

    def print_arithmetic(self, n=2):
        return f"\n{' ' * n}{self.__class__.__name__}({self.loss1.print_arithmetic(n + 2)} {self.loss2.print_arithmetic(n + 2)}\n{' ' * n})"


class AddLoss(MultiLossOperator):
    """
    Apply the `add` operation to the two losses.

    Examples:
        >>> loss_cls_1 = ValueLoss(2)
        >>> loss_cls_2 = Parameter("x")
        >>> loss_cls = loss_cls_1 + loss_cls_2  # equals to AddLoss(loss_cls_1, loss_cls_2)
        >>> print(loss_cls)
        x + 2
        >>> loss = loss_cls.eval({"x": 3})
        >>> print(loss)
        5

    """

    def __init__(self, loss1, loss2):
        super(AddLoss, self).__init__(loss1, loss2)

    @property
    def _symbol(self):
        return self.loss1._symbol + self.loss2._symbol

    def _get_eval(self, variables, **kwargs):
        loss1, loss2, variables = super()._get_eval(variables, **kwargs)
        return loss1 + loss2, variables


class SubLoss(MultiLossOperator):
    """
    Apply the `sub` operation to the two losses.

    Examples:
        >>> loss_cls_1 = ValueLoss(2)
        >>> loss_cls_2 = Parameter("x")
        >>> loss_cls = loss_cls_1 - loss_cls_2  # equals to SubLoss(loss_cls_1, loss_cls_2)
        >>> print(loss_cls)
        2 - x
        >>> loss = loss_cls.eval({"x": 4})
        >>> print(loss)
        -2
        >>> loss_cls = loss_cls_2 - loss_cls_1  # equals to SubLoss(loss_cls_2, loss_cls_1)
        >>> print(loss_cls)
        x - 2
        >>> loss = loss_cls.eval({"x": 4})
        >>> print(loss)
        2

    """

    def __init__(self, loss1, loss2):
        super(SubLoss, self).__init__(loss1, loss2)

    @property
    def _symbol(self):
        return self.loss1._symbol - self.loss2._symbol

    def _get_eval(self, variables, **kwargs):
        loss1, loss2, variables = super()._get_eval(variables, **kwargs)
        return loss1 - loss2, variables


class MulLoss(MultiLossOperator):
    """
    Apply the `mul` operation to the two losses.

    Examples:
        >>> loss_cls_1 = ValueLoss(2)
        >>> loss_cls_2 = Parameter("x")
        >>> loss_cls = loss_cls_1 * loss_cls_2  # equals to MulLoss(loss_cls_1, loss_cls_2)
        >>> print(loss_cls)
        2 x
        >>> loss = loss_cls.eval({"x": 4})
        >>> print(loss)
        8

    """

    def __init__(self, loss1, loss2):
        super(MulLoss, self).__init__(loss1, loss2)

    @property
    def _symbol(self):
        return self.loss1._symbol * self.loss2._symbol

    def _get_eval(self, variables, **kwargs):
        loss1, loss2, variables = super()._get_eval(variables, **kwargs)
        return loss1 * loss2, variables


class DivLoss(MultiLossOperator):
    """
    Apply the `div` operation to the two losses.

    Examples
        >>> loss_cls_1 = ValueLoss(2)
        >>> loss_cls_2 = Parameter("x")
        >>> loss_cls = loss_cls_1 / loss_cls_2  # equals to DivLoss(loss_cls_1, loss_cls_2)
        >>> print(loss_cls)
        \\frac{2}{x}
        >>> loss = loss_cls.eval({"x": 4})
        >>> print(loss)
        0.5
        >>> loss_cls = loss_cls_2 / loss_cls_1  # equals to DivLoss(loss_cls_2, loss_cls_1)
        >>> print(loss_cls)
        \\frac{x}{2}
        >>> loss = loss_cls.eval({"x": 4})
        >>> print(loss)
        2.0

    """

    def __init__(self, loss1, loss2):
        super(DivLoss, self).__init__(loss1, loss2)

    @property
    def _symbol(self):
        return self.loss1._symbol / self.loss2._symbol

    def _get_eval(self, variables, **kwargs):
        loss1, loss2, variables = super()._get_eval(variables, **kwargs)
        return loss1 / loss2, variables


class SingleLossOperator(AbstractLoss):
    def __init__(self, loss1):
        input_var = []

        if isinstance(loss1, AbstractLoss):
            input_var = deepcopy(loss1.input_var)
        elif isinstance(loss1, numbers.Number):
            loss1 = ValueLoss(loss1)
        else:
            raise ValueError()

        super(SingleLossOperator, self).__init__(input_var)

        self.loss1 = loss1

    def train(self, x_dict={}, **kwargs):
        return self.loss1.train(x_dict, **kwargs)

    def test(self, x_dict={}, **kwargs):
        return self.loss1.test(x_dict, **kwargs)

    def print_arithmetic(self, n=2):
        return f"\n{' ' * n}{self.__class__.__name__}({self.loss1.print_arithmetic(n + 2)}\n{' ' * n})"


class NegLoss(SingleLossOperator):
    """
    Apply the `neg` operation to the loss.

    Examples:
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

    def _get_eval(self, variables={}, **kwargs):
        loss, variables = self.loss1._get_eval(variables, **kwargs)
        return -loss, variables


class AbsLoss(SingleLossOperator):
    """
    Apply the `abs` operation to two losses.

    Examples:
        >>> import torch
        >>> from pixyz.distributions import Normal
        >>> from pixyz.losses import LogProb
        >>> p = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
        ...            features_shape=[10])
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

    def _get_eval(self, variables={}, **kwargs):
        loss, variables = self.loss1._get_eval(variables, **kwargs)
        return loss.abs(), variables


class BatchMean(SingleLossOperator):
    """
    Average a loss class over given batch data.

    .. math::

        \mathbb{E}_{p_{data}(x)}[\mathcal{L}(x)] \approx \frac{1}{N}\sum_{i=1}^N \mathcal{L}(x_i),

    where :math:`x_i \sim p_{data}(x)` and :math:`\mathcal{L}` is a loss function.

    Examples:
        >>> import torch
        >>> from pixyz.distributions import Normal
        >>> from pixyz.losses import LogProb
        >>> p = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
        ...            features_shape=[10])
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

    def _get_eval(self, variables={}, **kwargs):
        loss, variables = self.loss1._get_eval(variables, **kwargs)
        return loss.mean(), variables


class BatchSum(SingleLossOperator):
    """
    Summation a loss class over given batch data.

    .. math::

        \sum_{i=1}^N \mathcal{L}(x_i),

    where :math:`x_i \sim p_{data}(x)` and :math:`\mathcal{L}` is a loss function.

    Examples:
        >>> import torch
        >>> from pixyz.distributions import Normal
        >>> from pixyz.losses import LogProb
        >>> p = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.), var=["x"],
        ...            features_shape=[10])
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

    def _get_eval(self, variables={}, **kwargs):
        loss, variables = self.loss1._get_eval(variables, **kwargs)
        return loss.sum(), variables


class Expectation(Loss):
    r"""
    Expectation of a given function (Monte Carlo approximation).

    .. math::

        \mathbb{E}_{p(x)}[f(x)] \approx \frac{1}{L}\sum_{l=1}^L f(x_l),

    where :math:`x_l \sim p(x)`.

    Note that :math:`f` doesn't need to be able to sample, which is known as the law of the unconscious statistician
    (LOTUS).

    Therefore, in this class, :math:`f` is assumed to :attr:`pixyz.Loss`.

    Examples:
        >>> import torch
        >>> from pixyz.distributions import Normal
        >>> from pixyz.losses import LogProb
        >>> q = Normal(loc="x", scale=torch.tensor(1.), var=["z"], cond_var=["x"],
        ...            features_shape=[10]) # q(z|x)
        >>> p = Normal(loc="z", scale=torch.tensor(1.), var=["x"], cond_var=["z"],
        ...            features_shape=[10]) # p(x|z)
        >>> loss_cls = LogProb(p).expectation(q) # equals to Expectation(q, LogProb(p))
        >>> print(loss_cls)
        \mathbb{E}_{p(z|x)} \left[\log p(x|z) \right]
        >>> sample_x = torch.randn(2, 10) # Psuedo data
        >>> loss = loss_cls.eval({"x": sample_x})
        >>> print(loss) # doctest: +SKIP
        tensor([-12.8181, -12.6062])
        >>> loss_cls = LogProb(p).expectation(q, sample_shape=(5,)) # equals to Expectation(q, LogProb(p))
        >>> loss = loss_cls.eval({"x": sample_x})
        >>> print(loss) # doctest: +SKIP

    """

    def __init__(self, p, f, input_var=None, sample_shape=torch.Size([1])):
        if input_var is None:
            input_var = list(set(p.input_var) | set(f.input_var) - set(p.var))
        self._f = f
        self.sample_shape = torch.Size(sample_shape)

        super().__init__(p=p, input_var=input_var)

    @property
    def _symbol(self):
        p_text = "{" + self.p.prob_text + "}"
        return sympy.Symbol("{{E}}_{} \\left[{} \\right]".format(p_text, self._f.loss_text))

    def _get_eval(self, variables, **kwargs):
        samples_dicts = [self.p.sample(variables, reparam=True, return_all=True) for i in range(self.sample_shape.numel())]
        loss_and_dicts = [self._f.eval(samples_dict, return_dict=True, **kwargs) for
                          samples_dict in samples_dicts]  # TODO: eval or _get_eval
        losses = [loss for loss, loss_sample_dict in loss_and_dicts]
        # sum over sample_shape
        loss = torch.stack(losses).mean(dim=0)
        samples_dicts[0].update(loss_and_dicts[0][1])

        return loss, samples_dicts[0]

    def print_arithmetic(self, n=2):
        return f"\n{' ' * n}{self.__class__.__name__}({self.input_var})"
