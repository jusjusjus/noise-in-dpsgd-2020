
from torch import nn


class Optimizable(nn.Module):

    _opt_err_str = """optimization function not initialized
    Call first `Optimizable.init_optimizer(...)`"""

    def __init__(self, **kwargs):
        super().__init__()
        self.config_params = []
        for k, v in kwargs.items():
            self.config_params.append(k)
            setattr(self, k, v)

    def config(self):
        return {
            k: getattr(self, k)
            for k in self.config_params
        }

    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_dict(self):
        return {
            'config': self.config(),
            'params': self.state_dict()
        }

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        assert all(k in state_dict for k in ('config', 'params')), f"""
        state dict of wrong format {state_dict}"""
        instance = cls(**state_dict['config'])
        instance.load_state_dict(state_dict['params'], **kwargs)
        return instance

    def init_optimizer(self, opt_fn, *args, **kwargs):
        self.opt = opt_fn(self.parameters(), *args, **kwargs)

    def zero_grad(self):
        try:
            return self.opt.zero_grad()
        except AttributeError:
            raise AttributeError(self._opt_err_str)

    def step(self):
        try:
            return self.opt.step()
        except AttributeError:
            raise AttributeError(self._opt_err_str)


