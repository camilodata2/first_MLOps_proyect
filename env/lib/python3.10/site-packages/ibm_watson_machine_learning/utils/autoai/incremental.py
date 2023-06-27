#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2022- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


def calculate_cum_limits(scores):
    def lower_limit(values):
        mean = np.mean(values)
        std = np.std(values)
        return mean - std

    def upper_limit(values):
        mean = np.mean(values)
        std = np.std(values)
        return mean + std

    n = len(scores) + 1
    lower_limits = [lower_limit(scores[:i]) for i in range(1, n)]
    upper_limits = [upper_limit(scores[:i]) for i in range(1, n)]

    return lower_limits, upper_limits


def plot_subcurve(fig, ax, values, x=None, color='b', title='Learning Curve', xlabel='Training batches', ylabel='Score',
                  refresh=True):
    if x is None:
        x = list(range(0, len(values)))

    lower_limits, upper_limits = calculate_cum_limits(values)
    ax.cla()
    ax.grid()
    ax.plot(x, values, "o-")
    ax.fill_between(
        x,
        lower_limits,
        upper_limits,
        alpha=0.1,
        color=color,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if refresh:
        display(fig)
        clear_output(wait=True)


def plot_learning_curve(fig, axes, scores, fit_times=None):
    if isinstance(axes, np.ndarray) and fit_times is not None:
        # plot score vs. batch number
        plot_subcurve(fig, axes[0], scores)

        if len(fit_times) < len(scores):
            x = list(range(len(scores) - len(fit_times), len(scores)))
            scores_aligned = scores[len(scores) - len(fit_times):]
        else:
            x = list(range(0, len(scores)))
            scores_aligned = scores

        # plot fit_time vs. batch number
        plot_subcurve(fig, axes[1], fit_times, x, color='g', title='Scalability of the model',
                      xlabel='Training batches', ylabel='Fit time')

        if len(axes) == 3:
            # plot fit_time vs. score
            fit_times_np = np.array(fit_times)
            fit_times_cumsum = np.cumsum(fit_times_np)
            plot_subcurve(fig, axes[2], scores_aligned, x=fit_times_cumsum, color='r', title='Performance of the model',
                          xlabel='Fit time', ylabel='Score')
    else:
        # plot score vs. batch number
        plot_subcurve(fig, axes, scores)
