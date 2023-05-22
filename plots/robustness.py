import json
from pathlib import Path
from typing import List

import fire
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.pyplot import rc

PERCENTAGES = [2, 10, 20, 30, 40, 50, 60, 70, 80, 90]

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def plot_robustness_histogram(json_path: str,
                              dimension_key: str = "ph_dim_losses_based",
                              subset_key_stem: str = "ph_dim_subset_based_",
                              percentages: List[int] = PERCENTAGES,
                              title: str = "Robustness",
                              scale: str = "linear",
                              width: int = 5,
                              save: bool = False,
                              y_legend: bool = True):

    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"{str(json_path)} not found.")

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    experiments = [k for k in results.keys() if dimension_key in results[k].keys()]

    logger.info(f"Found {len(experiments)} converged experiments")
    mean_error_tab = []
    standard_deviation_tab = []

    for perc in percentages:

        dim_key = subset_key_stem + str(perc)
        relative_error_tab = []

        for exp in experiments:

            reference_dim = results[exp][dimension_key]

            assert dim_key in results[exp].keys(), results[exp].keys()
            subset_dim = results[exp][dim_key]

            relative_error = np.abs((subset_dim - reference_dim) / reference_dim)
            relative_error_tab.append(relative_error)

        relative_error_tab = np.array(relative_error_tab)
        mean = relative_error_tab.mean()
        std = relative_error_tab.std()
        mean_error_tab.append(mean)
        standard_deviation_tab.append(std)

    percentages = np.array(percentages)

    plt.bar(x=percentages, height=mean_error_tab,
            yerr=standard_deviation_tab, capsize=4,
            width=width)
    plt.title(title)
    plt.xlabel(r"\textbf{Data proportion}")
    if y_legend:
        plt.ylabel(r"\textbf{Relative error}")
    plt.legend()

    plt.ylim(0., 0.085)
    plt.xlim(0.)

    if scale == "log":
        plt.yscale("log")

    logger.info(f"Means relative error: {mean_error_tab}")

    if save:

        output_path = json_path.parent / f"{json_path.parent.name}_robustness.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving figure in {str(output_path)}")
        plt.savefig(str(output_path))


def plot_in_line(json1: str,
                 json2: str,
                 output_dir: str):

    plt.figure(figsize=(8, 4))

    plt.subplot(121)
    plot_robustness_histogram(json1,
                              title=r"\textbf{MNIST}", save=False)

    plt.subplot(122)
    plot_robustness_histogram(json2,
                              title=r"\textbf{CHD}", save=False,
                              y_legend=False)

    output_path = Path(output_dir) / "robustness.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving figure in {str(output_path)}")
    plt.savefig(str(output_path))


if __name__ == "__main__":
    fire.Fire({"one": plot_robustness_histogram,
               "line": plot_in_line})
