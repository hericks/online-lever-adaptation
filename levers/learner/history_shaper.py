from typing import List

import torch
import torch.nn as nn

class HistoryShaper:

    def __init__(self, hs_net: nn.Module):
        """
        Simple wrapper for history shaping network. 
        """
        self.net = hs_net

    def reset(self, params: List[torch.Tensor]):
        """
        Resets the parameters of the history shaper to those passed in
        `params`.
        """
        with torch.no_grad():
            for i, param in enumerate(self.net.parameters()):
                param.set_(params[i].clone().detach())
