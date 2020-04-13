import sympy
import torch
from torch.distributions import kl_divergence

from .losses import Loss


class KullbackLeibler(Loss):

    def __init__(self, p, q, input_var=None, dim=None):
        self.dim = dim
        super().__init__(p, q, input_var)

    @property
    def _symbol(self):
        return sympy.Symbol("D_{{KL}} \\left[{}||{} \\right]".format(self.p.prob_text, self.q.prob_text))

    def _get_eval(self, variables, **kwargs):
        if (not hasattr(self.p, 'distribution_torch_class')) or (not hasattr(self.q, 'distribution_torch_class')):
            raise ValueError(
                "Divergence between these two distributions cannot be evaluated, "
                f"got {self.p.distribution_name} and {self.q.distribution_name}."
            )
        input_p = variables.get_variables(self.p.input_var)
        self.p.set_dist(input_p)

        input_q = variables.get_variables(self.q.input_var)
        self.q.set_dist(input_q)

        divergence = kl_divergence(self.p.dist, self.q.dist)

        if len(self.p.cond_var) > 0:
            latent_var = self.p.get_sample(reparam=True)
            variables.update(latent_var)

        if len(self.q.cond_var) > 0:
            latent_var = self.q.get_sample(reparam=True)
            variables.update(latent_var)

        if self.dim:
            divergence = torch.sum(divergence, dim=self.dim)
            return divergence, variables

        dim_list = list(torch.arange(divergence.dim()))
        divergence = torch.sum(divergence, dim=dim_list[1:])
        return divergence, variables

    def print_arithmetic(self, n=0):
        return f"\n{' ' * n}{self.__class__.__name__}({self.p.name}, {self.q.name})"
