import torch


def ELBO(p, q, input_var=None, sample_shape=torch.Size([1])):
    r"""
    The evidence lower bound (Monte Carlo approximation).

    .. math::

        \mathbb{E}_{q(z|x)}\left[\log \frac{p(x,z)}{q(z|x)}\right] \approx \frac{1}{L}\sum_{l=1}^L \log p(x, z_l),
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
    >>> q = NormalQ(var=["z"], cond_var=["x"], features_shape=[64]) # q(z|x)
    >>> class NormalP(Normal):
    ...     def forward(self, z, **kwargs):
    ...         return {'loc': z, 'scale': torch.ones(1, 64)}
    >>> p = NormalP(var=["x"], cond_var=["z"], features_shape=[64]) # p(x|z)
    >>> loss_cls = ELBO(p, q)
    >>> print(loss_cls)
    \mathbb{E}_{p(z|x)} \left[\log p(x|z) - \log p(z|x) \right]
    >>> loss = loss_cls.eval({"x": torch.randn(1, 64)})
    """
    loss = (p.log_prob() - q.log_prob()).expectation(q, input_var, sample_shape=sample_shape)
    return loss
