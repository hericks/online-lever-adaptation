from typing import List, Tuple


def generate_binary_patterns(length: int) -> List[Tuple[int]]:
    """
    Recursive function returning list of all binary tuples of length `length`.
    """
    if length == 0:
        return [()]
    else:
        tails = generate_binary_patterns(length-1)
        return [(0,) + tail for tail in tails] + [(1,) + tail for tail in tails]