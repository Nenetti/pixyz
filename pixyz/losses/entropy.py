import sympy
import torch

from pixyz.losses.losses import Loss
from pixyz.losses.divergences import KullbackLeibler


def Entropy(p, input_var=None, analytical=True, sample_shape=torch.Size([1])):
    r"""
    Entropy (Analytical or Monte Carlo approximation).

    .. math::

        H(p) &= -\mathbb{E}_{p(x)}[\log p(x)] \qquad \text{(analytical)}\\
        &\approx -\frac{1}{L}\sum_{l=1}^L \log p(x_l), \quad \text{where} \quad x_l \sim p(x) \quad \text{(MC approximation)}.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> class NormalP(Normal):
    ...     def forward(self, **kwargs):
    ...         return {'loc': torch.zeros(1, 64), 'scale': torch.ones(1, 64)}
    >>> p = NormalP(var=["x"], features_shape=[64])
    >>> loss_cls = Entropy(p, analytical=True)
    >>> print(loss_cls)
    H \left[ {p(x)} \right]
    >>> loss_cls.eval()
    tensor([90.8121])
    >>> loss_cls = Entropy(p, analytical=False, sample_shape=[10])
    >>> print(loss_cls)
    - \mathbb{E}_{p(x)} \left[\log p(x) \right]
    >>> loss_cls.eval() # doctest: +SKIP
    tensor([90.5991])
    """
    if analytical:
        loss = AnalyticalEntropy(p, input_var=input_var)
    else:
        if input_var is None:
            input_var = p.input_var
        loss = -p.log_prob().expectation(p, input_var, sample_shape=sample_shape)
    return loss


class AnalyticalEntropy(Loss):
    def __init__(self, p, input_var=None):
        if input_var is None:
            _input_var = p.input_var.copy()
        else:
            _input_var = input_var
        super().__init__(_input_var)
        self.p = p

    @property
    def _symbol(self):
        p_text = "{" + self.p.prob_text + "}"
        return sympy.Symbol(f"H \\left[ {p_text} \\right]")

    def forward(self, x_dict, **kwargs):
        if not hasattr(self.p, 'get_distribution_torch_class'):
            raise ValueError("Entropy of this distribution cannot be evaluated, "
                             "got %s." % self.p.distribution_name)

        entropy = self.p.get_entropy(x_dict)

        return entropy, x_dict


def CrossEntropy(p, q, input_var=None, analytical=False, sample_shape=torch.Size([1])):
    r"""
    Cross entropy, a.k.a., the negative expected value of log-likelihood (Monte Carlo approximation or Analytical).

    .. math::

        H(p,q) &= -\mathbb{E}_{p(x)}[\log q(x)] \qquad \text{(analytical)}\\
        &\approx -\frac{1}{L}\sum_{l=1}^L \log q(x_l), \quad \text{where} \quad x_l \sim p(x) \quad \text{(MC approximation)}.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> class NormalP(Normal):
    ...     def forward(self, **kwargs):
    ...         return {'loc': torch.zeros(1, 64), 'scale': torch.ones(1, 64)}
    >>> p = NormalP(var=["x"], features_shape=[64], name="p")
    >>> class NormalQ(Normal):
    ...     def forward(self, **kwargs):
    ...         return {'loc': torch.ones(1, 64), 'scale': torch.ones(1, 64)}
    >>> q = NormalQ(var=["x"], features_shape=[64], name="q")
    >>> loss_cls = CrossEntropy(p, q, analytical=True)
    >>> print(loss_cls)
    D_{KL} \left[p(x)||q(x) \right] + H \left[ {p(x)} \right]
    >>> loss_cls.eval()
    tensor([122.8121])
    >>> loss_cls = CrossEntropy(p, q, analytical=False, sample_shape=[10])
    >>> print(loss_cls)
    - \mathbb{E}_{p(x)} \left[\log q(x) \right]
    >>> loss_cls.eval() # doctest: +SKIP
    tensor([123.2192])
    """
    if analytical:
        loss = Entropy(p) + KullbackLeibler(p, q)
    else:
        if input_var is None:
            input_var = list(set(p.input_var + q.input_var) - set(p.var))

        loss = -q.log_prob().expectation(p, input_var, sample_shape=sample_shape)
    return loss


def StochasticReconstructionLoss(encoder, decoder, input_var=None, sample_shape=torch.Size([1])):
    r"""
    Reconstruction Loss (Monte Carlo approximation).

    .. math::

        -\mathbb{E}_{q(z|x)}[\log p(x|z)] \approx -\frac{1}{L}\sum_{l=1}^L \log p(x|z_l),
         \quad \text{where} \quad z_l \sim q(z|x).

    Note:
        This class is a special case of the :attr:`Expectation` class.

    Examples
    --------
    >>> import torch
    >>> from pixyz.distributions import Normal
    >>> class NormalQ(Normal):
    ...     def forward(self, x, **kwargs):
    ...         return {'loc': x, 'scale': torch.ones(1, 64)}
    >>> q = NormalQ(var=["z"], cond_var=["x"], features_shape=[64], name="q") # q(z|x)
    >>> class NormalP(Normal):
    ...     def forward(self, z, **kwargs):
    ...         return {'loc': z, 'scale': torch.ones(1, 64)}
    >>> p = NormalP(var=["x"], cond_var=["z"], features_shape=[64], name="p") # p(x|z)
    >>> loss_cls = StochasticReconstructionLoss(q, p)
    >>> print(loss_cls)
    - \mathbb{E}_{q(z|x)} \left[\log p(x|z) \right]
    >>> loss = loss_cls.eval({"x": torch.randn(1,64)})
    """
    if input_var is None:
        input_var = encoder.input_var

    if not (set(decoder.var) <= set(input_var)):
        raise ValueError("Variable {} (in the `{}` class) is not included"
                         " in `input_var` of the `{}` class.".format(decoder.var,
                                                                     decoder.__class__.__name__,
                                                                     encoder.__class__.__name__))

    loss = -decoder.log_prob().expectation(encoder, input_var, sample_shape=sample_shape)
    return loss
