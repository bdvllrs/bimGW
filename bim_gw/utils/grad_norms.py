import numpy as np
import pandas as pd


class GradNormLogger:
    def __init__(self):
        self.grad_norms = dict()

    def log(self, grad_norms):
        for key, val in grad_norms.items():
            [loss_name, model] = key.split("@")
            if model not in self.grad_norms:
                self.grad_norms[model] = {}
            if loss_name not in self.grad_norms[model]:
                self.grad_norms[model][loss_name] = []
            self.grad_norms[model][loss_name].append(val.detach().cpu().item())

    def values(self, window_size=1):
        norms = {}
        for model in self.grad_norms.keys():
            if model not in norms:
                norms[model] = {}
            for loss_name in self.grad_norms[model].keys():
                if window_size == 1:
                    norms[model][loss_name] = self.grad_norms[model][loss_name][-1]
                else:
                    norms[model][loss_name] = (
                                np.convolve(self.grad_norms[model][loss_name][-window_size:], np.ones(window_size),
                                            'valid') / window_size)[-1]

        return pd.DataFrame(norms)
