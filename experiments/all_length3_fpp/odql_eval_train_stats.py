from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from levers.helpers import generate_binary_patterns


def save_smooth_mean_evals(mean_evals: np.array, out_path: str):
    epochs = np.arange(mean_evals.shape[1])
    plt.plot(epochs, mean_evals.mean(0))
    plt.fill_between(
        epochs,
        mean_evals.mean(0) - mean_evals.std(0),
        mean_evals.mean(0) + mean_evals.std(0),
        alpha=0.25,
    )
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    # Settings
    n_train_evals = 25
    n_epochs = 300

    # Save figures
    out_path = f"results/train_figures/odql"
    mean_evals_template = "ODQL-mean-evals-{pattern}.png"

    patterns = generate_binary_patterns(3)
    for pattern_id, pattern in enumerate(combinations(patterns, 4)):
        print(f"Pattern: {pattern} | {pattern_id+1:2d} / 70")

        mean_evals = np.zeros((n_train_evals, n_epochs))

        for train_id in range(n_train_evals):
            try:
                train_stats = torch.load(
                    f"train_stats/odql/ODQL-{pattern}-{train_id}.pickle"
                )
                mean_evals[train_id, :] = np.array(train_stats["mean_eval"])
            except:
                print(
                    f"Something went wrong with reading in data for: {pattern}-{train_id}."
                )

        save_smooth_mean_evals(
            mean_evals,
            os.path.join(out_path, mean_evals_template.format(pattern=pattern)),
        )
