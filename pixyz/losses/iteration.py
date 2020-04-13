from copy import deepcopy
import sympy

from .losses import Loss, AbstractLoss
from ..utils import get_dict_values
from ..variables import Variables


class IterativeLoss(Loss):
    """
    Iterative loss.
    This class allows implementing an arbitrary model which requires iteration.

    .. math::
        \mathcal{L} = \sum_{t=1}^{T}\mathcal{L}_{step}(x_t, h_t),

        where :math:`x_t = f_{slice\_step}(x, t)`.

    Examples:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from pixyz.distributions import Normal, Bernoulli, Deterministic
        >>>
        >>> # Set distributions
        >>> x_dim = 128
        >>> z_dim = 64
        >>> h_dim = 32
        >>>
        >>> # p(x|z,h_{prev})
        >>> class Decoder(Bernoulli):
        ...     def __init__(self):
        ...         super().__init__(cond_var=["z", "h_prev"], var=["x"], name="p")
        ...         self.fc = torch.nn.Linear(z_dim + h_dim, x_dim)
        ...     def forward(self, z, h_prev):
        ...         return {"probs": torch.sigmoid(self.fc(torch.cat((z, h_prev), dim=-1)))}
        ...
        >>> # q(z|x,h_{prev})
        >>> class Encoder(Normal):
        ...     def __init__(self):
        ...         super().__init__(cond_var=["x", "h_prev"], var=["z"], name="q")
        ...         self.fc_loc = torch.nn.Linear(x_dim + h_dim, z_dim)
        ...         self.fc_scale = torch.nn.Linear(x_dim + h_dim, z_dim)
        ...     def forward(self, x, h_prev):
        ...         xh = torch.cat((x, h_prev), dim=-1)
        ...         return {"loc": self.fc_loc(xh), "scale": F.softplus(self.fc_scale(xh))}
        ...
        >>> # f(h|x,z,h_{prev}) (update h)
        >>> class Recurrence(Deterministic):
        ...     def __init__(self):
        ...         super().__init__(cond_var=["x", "z", "h_prev"], var=["h"], name="f")
        ...         self.rnncell = torch.nn.GRUCell(x_dim + z_dim, h_dim)
        ...     def forward(self, x, z, h_prev):
        ...         return {"h": self.rnncell(torch.cat((z, x), dim=-1), h_prev)}
        >>>
        >>> p = Decoder()
        >>> q = Encoder()
        >>> f = Recurrence()
        >>>
        >>> # Set the loss class
        >>> step_loss_cls = p.log_prob().expectation(q * f).mean()
        >>> print(step_loss_cls)
        mean \left(\mathbb{E}_{p(h,z|x,h_{prev})} \left[\log p(x|z,h_{prev}) \right] \right)
        >>> loss_cls = IterativeLoss(step_loss=step_loss_cls,
        ...                          series_var=["x"], update_value={"h": "h_prev"})
        >>> print(loss_cls)
        \sum_{t=1}^{t_{max}} mean \left(\mathbb{E}_{p(h,z|x,h_{prev})} \left[\log p(x|z,h_{prev}) \right] \right)
        >>>
        >>> # Evaluate
        >>> x_sample = torch.randn(30, 2, 128) # (timestep_size, batch_size, feature_size)
        >>> h_init = torch.zeros(2, 32) # (batch_size, h_dim)
        >>> loss = loss_cls.eval({"x": x_sample, "h_prev": h_init})
        >>> print(loss) # doctest: +SKIP
        tensor(-2826.0906, grad_fn=<AddBackward0>
    """

    def __init__(self, step_loss, max_iter=None, input_var=None, series_var=None, update_value={}, slice_step=None, timestep_var=["t"]):
        self.step_loss = step_loss
        self.max_iter = max_iter
        self.update_value = update_value
        self.timestep_var = timestep_var
        self.timpstep_symbol = sympy.Symbol(self.timestep_var[0])

        if (series_var is None) and (max_iter is None):
            raise ValueError()

        self.slice_step = slice_step
        if self.slice_step:
            self.step_loss = self.step_loss.expectation(self.slice_step)

        if input_var is not None:
            self._input_var = input_var
        else:
            _input_var = []
            _input_var += deepcopy(self.step_loss.input_var)

            self._input_var = sorted(set(_input_var), key=_input_var.index)

            if slice_step:
                self._input_var.remove(timestep_var[0])  # delete a time-step variable from input_var

        self.series_var = series_var

    @property
    def _symbol(self):
        # TODO: naive implementation
        dummy_loss = sympy.Symbol("dummy_loss")
        if self.max_iter:
            max_iter = self.max_iter
        else:
            max_iter = sympy.Symbol(sympy.latex(self.timpstep_symbol) + "_{max}")

        _symbol = sympy.Sum(dummy_loss, (self.timpstep_symbol, 1, max_iter))
        _symbol = _symbol.subs({dummy_loss: self.step_loss._symbol})
        return _symbol

    def slice_step_fn(self, t, x):
        return {k: v[t] for k, v in x.items()}

    def _get_eval(self, variables, **kwargs):
        series_x_dict = variables.get_variables(self.series_var)
        updated_x_dict = variables.get_variables(list(self.update_value.values()))

        step_loss_sum = 0

        # set max_iter
        if self.max_iter:
            max_iter = self.max_iter
        else:
            max_iter = len(series_x_dict[self.series_var[0]])

        if "mask" in kwargs.keys():
            mask = kwargs["mask"].float()
        else:
            mask = None

        for t in range(max_iter):
            if self.slice_step:
                variables.update({self.timestep_var[0]: t})
            else:
                # update series inputs & use slice_step_fn
                variables.update(self.slice_step_fn(t, series_x_dict))

            # evaluate
            step_loss, samples = self.step_loss.eval(variables, return_dict=True)
            variables.update(samples)
            if mask is not None:
                step_loss *= mask[t]
            step_loss_sum += step_loss

            # update
            for key, value in self.update_value.items():
                variables.update({value: variables[key]})

        loss = step_loss_sum

        # Restore original values
        variables.update(series_x_dict)
        variables.update(updated_x_dict)
        return loss, variables


class RecurrentLoss(AbstractLoss):

    def __init__(self, loss, max_iter, iter_var):
        self.loss = loss
        self.max_iter = max_iter
        self.iter = 0
        self.iter_var = iter_var
        self.iter_variables = Variables()

        _input_var = []
        _input_var += deepcopy(self.loss.input_var)

        self._input_var = sorted(set(_input_var), key=_input_var.index)

    def _get_eval(self, variables, **kwargs):
        self.iter += 1

        input_var = variables.get_variables(self._input_var)

        loss, samples = self.loss.eval(input_var, return_dict=True)

        variables.update(samples)
        return loss, variables

    def print_arithmetic(self, n=2):
        return f"\n{' ' * n}{self.__class__.__name__}({self.loss.print_arithmetic(n + 2)}\n{' ' * n})"


class SummativeRecurrentLoss(RecurrentLoss):
    """
    Iterative loss.
    This class allows implementing an arbitrary model which requires iteration.

    .. math::
        \mathcal{L} = \sum_{t=1}^{T}\mathcal{L}_{step}(x_t, h_t),

        where :math:`x_t = f_{slice\_step}(x, t)`.
    """

    def __init__(self, loss, max_iter, iter_var):
        super(SummativeRecurrentLoss, self).__init__(loss, max_iter, iter_var)

    @property
    def _symbol(self):
        return sympy.Symbol(f"\sum_{{t=0}}^{{{self.max_iter}}}{self.loss._symbol}")


class ProductiveRecurrentLoss(RecurrentLoss):
    """
    Iterative loss.
    This class allows implementing an arbitrary model which requires iteration.

    .. math::
        \mathcal{L} = \sum_{t=1}^{T}\mathcal{L}_{step}(x_t, h_t),

        where :math:`x_t = f_{slice\_step}(x, t)`.
    """

    def __init__(self, loss, max_iter, iter_var={}):
        super(ProductiveRecurrentLoss, self).__init__(loss, max_iter, iter_var)

    @property
    def _symbol(self):
        return sympy.Symbol(f"\prod_{{t=0}}^{{{self.max_iter}}}{self.loss._symbol}")

# class SummativeRecurrentLoss(AbstractLoss):
#     """
#     Iterative loss.
#     This class allows implementing an arbitrary model which requires iteration.
#
#     .. math::
#         \mathcal{L} = \sum_{t=1}^{T}\mathcal{L}_{step}(x_t, h_t),
#
#         where :math:`x_t = f_{slice\_step}(x, t)`.
#     """
#
#     def __init__(self, loss, max_iter, iter_var):
#         self.loss = loss
#         self.max_iter = max_iter
#         self.iter = 0
#         self.iter_var = iter_var
#         self.iter_variables = Variables()
#
#         _input_var = []
#         _input_var += deepcopy(self.loss.input_var)
#
#         self._input_var = sorted(set(_input_var), key=_input_var.index)
#
#     @property
#     def _symbol(self):
#         return sympy.Symbol(f"\sum_{{t=0}}^{{{self.max_iter}}}{self.loss._symbol}")
#
#     def _get_eval(self, variables, **kwargs):
#         self.iter += 1
#
#         input_var = variables.get_variables(self._input_var)
#
#         loss, samples = self.loss.eval(input_var, return_dict=True)
#
#         iter_var = samples.get_variables(self.iter_var).replace_dict_keys({self.iter_var: f"{self.iter_var}_{self.iter}"})
#         self.iter_variables.update(iter_var)
#         print(self.iter_variables.keys())
#         # samples =
#         samples = samples.delete_dict_values(self.iter_var)
#         variables = variables.delete_dict_values(self.iter_var)
#
#         variables.update(samples)
#         if self.iter == self.max_iter:
#             variables.update(self.iter_variables)
#             self.iter_variables.clear()
#             self.iter = 0
#         return loss, variables
#
#     def print_arithmetic(self, n=2):
#         return f"\n{' ' * n}{self.__class__.__name__}({self.loss.print_arithmetic(n + 2)}\n{' ' * n})"
#
#
# class ProductiveRecurrentLoss(AbstractLoss):
#     """
#     Iterative loss.
#     This class allows implementing an arbitrary model which requires iteration.
#
#     .. math::
#         \mathcal{L} = \sum_{t=1}^{T}\mathcal{L}_{step}(x_t, h_t),
#
#         where :math:`x_t = f_{slice\_step}(x, t)`.
#     """
#
#     def __init__(self, loss, max_iter, iter_var={}):
#         self.loss = loss
#         self.max_iter = max_iter
#         self.iter = 0
#
#         _input_var = []
#         _input_var += deepcopy(self.loss.input_var)
#
#         self._input_var = sorted(set(_input_var), key=_input_var.index)
#
#     @property
#     def _symbol(self):
#         return sympy.Symbol(f"\prod_{{t=0}}^{{{self.max_iter}}}{self.loss._symbol}")
#
#     def _get_eval(self, variables, **kwargs):
#         input_var = variables.get_variables(self._input_var)
#
#         loss, samples = self.loss.eval(input_var, return_dict=True)
#
#         variables.update(samples)
#         self.iter += 1
#         return loss, variables
#
#     def print_arithmetic(self, n=2):
#         return f"\n{' ' * n}{self.__class__.__name__}({self.loss.print_arithmetic(n + 2)}\n{' ' * n})"
