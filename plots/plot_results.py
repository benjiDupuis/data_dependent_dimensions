import json
from pathlib import Path

import fire
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from matplotlib import rc

MARKERS = ["o", "^", "s", "+", "D", "v", "*", "p"]

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def plot_from_json(json_path: str,
                   generalization_key: str = "loss_gap",
                   dimension_key: str = "ph_dim_losses_based",
                   title: str = "Dimension VS generalization error",
                   scale: str = "linear",
                   save: bool = True,
                   colorbar: bool = True
                   ):

    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"{str(json_path)} not found.")

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    experiments = [k for k in results.keys() if
                   generalization_key in results[k].keys() and
                   dimension_key in results[k].keys()]

    logger.info(f"Found {len(experiments)} converged experiments")

    color_map = plt.cm.get_cmap("viridis")

    all_bs = list(set([results[k]["batch_size"] for k in experiments]))

    for b_idx in range(len(all_bs)):

        b = all_bs[b_idx]
        b_experiments = [k for k in experiments if results[k]["batch_size"] == b]

        dim_tab = [results[k][dimension_key] for k in b_experiments]
        gen_tab = [results[k][generalization_key] for k in b_experiments]
        lr_tab = [results[k]["learning_rate"] for k in b_experiments]

        # logger.debug(f"Number of batch size {b}: {len(dim_tab)}")

        sc = plt.scatter(
            np.array(gen_tab),
            np.array(dim_tab),
            c=np.array(lr_tab),
            marker=MARKERS[b_idx],
            label=str(b),
            cmap=color_map
        )

    if scale == "log":
        plt.xscale("log")
        plt.yscale("log")

    plt.xlabel(r"\textbf{Loss gap}", loc="right")
    plt.ylabel(r'$\textbf{dim}_{\textbf{PH}}$')

    if colorbar:
        cbar = plt.colorbar(sc)
        cbar.set_label(r"Learning rate")

    plt.grid()
    # plt.legend()

    plt.title(title, loc="left")

    if save:
        output_path = json_path.parent / f"{json_path.parent.name}_plot.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving figure in {str(output_path)}")
        plt.savefig(str(output_path))


def plot_in_column(output_dir: str,
                   json1: str,
                   json2: str,
                   json3: str = None,
                   title: str = "CHD"):

    plt.figure(figsize=(5, 7))

    plt.subplot(211)
    plot_from_json(json1, colorbar=False, save=False,
                   title=r"FCN-$7$ on CHD")

    plt.subplot(212)
    plot_from_json(json2, colorbar=False, save=False,
                   title=r"FCN-$5$ on CHD")

    if json3 is not None:
        plt.subplot(313)
        plot_from_json(json3, colorbar=False, save=False,
                       title=r"AlexNet on CIFAR$10$")

    output_path = Path(output_dir) / "CHD.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving figure in {str(output_path)}")
    plt.savefig(str(output_path))


if __name__ == "__main__":
    fire.Fire({"one": plot_from_json,
               "column": plot_in_column})
