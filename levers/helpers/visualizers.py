from typing import Optional

import matplotlib.pyplot as plt

import numpy as np


def visualize_xplay_matrix(
    matrix: np.array,
    out_path: str,
    vmin: float,
    vmax: float,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    plt.rcParams.update({"font.size": 25})
    plt.matshow(matrix, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 18.5, forward=True)
    plt.savefig(out_path)
    plt.close(fig)
