from typing import List, Tuple

import torch.nn as nn


def generate_binary_patterns(length: int) -> List[Tuple[int]]:
    """
    Recursive function returning list of all binary tuples of length `length`.
    """
    if length == 0:
        return [()]
    else:
        tails = generate_binary_patterns(length-1)
        return [(0,) + tail for tail in tails] + [(1,) + tail for tail in tails]

def n_total_parameters(net: nn.Module) -> int:
    """
    Returns the total number of parameters in network `net`. 
    """
    return sum(p.nume() for p in net.parameters())