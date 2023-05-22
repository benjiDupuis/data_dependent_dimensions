import datetime
import json
import os
from pathlib import Path

import fire
import numpy as np
import wandb
from loguru import logger
from pydantic import BaseModel

from PHDim.train_risk_analysis import main as risk_analysis


# Uncomment for wandb logging
# def reset_wandb_env():
#     exclude = {
#         "WANDB_PROJECT",
#         "WANDB_ENTITY",
#         "WANDB_API_KEY",
#     }
#     for k, v in os.environ.items():
#         if k.startswith("WANDB_") and k not in exclude:
#             del os.environ[k]


class AnalysisOptions(BaseModel):

    """
    All hyperparameters of the experiement are defined here
    """

    iterations: int = 10000000000  # Maximum authorized number of iterations
    log_weights: bool = True  # Whether we want to save final weights of the experiment
    batch_size_eval: int = 5000  # batch size used for evaluation
    lrmin: float = 0.005  # minimum learning rate in teh experiment
    lrmax: float = 0.1  # maximum learning rate in the experiment
    bs_min: int = 32  # minimum batch size in the experiment
    bs_max: int = 256  # maximum batch sie in the experiment
    eval_freq: int = 3000  # at which frequency we evaluate the model (training and validation sets)
    dataset: str = "cifar10"  # dataset we use
    data_path: str = "~/data/"  # where to find the data
    model: str = "alexnet"  # model, currently supported: ["fc", "alexnet", "vgg", "lenet"]
    save_folder: str = "./results"  # Where to save the results
    depth: int = 5  # depth of the network (for FCNN)
    width: int = 200  # width of the network (for FCNN)
    optim: str = "SGD"  # Optimizer
    min_points: int = 1000  # minimum number of points used to compute the PH dimension
    num_exp_lr: int = 6  # Number of batch sizes we use
    num_exp_bs: int = 6  # Number of learning rates we use
    compute_dimensions: bool = True  # whether or not we compute the PH dimensions
    project_name: str = "ph_dim_mnist"  # project name for WANDB logging
    initial_weights: str = None  # Initial weights if they exist, always none in our work
    ripser_points: int = 5000  # Maximum number of points used to compute the PH dimension
    seed: int = 1234  # seed
    jump: int = 20  # number of finite sets drawn to compute the PH dimension, see https://arxiv.org/abs/2111.13171v1
    additional_dimensions: bool = False  # whether or not compute the ph dimensions used in the robustness experiment
    data_proportion: float = 1. # Proportion of data to use (between 0 and 1), used for pytests

    def __call__(self):

        save_folder = Path(self.save_folder)

        exp_folder = save_folder / str(datetime.datetime.now()).replace(" ", "_").replace(":", "_")
        exp_folder.mkdir(parents=True, exist_ok=True)

        log_file = exp_folder / "parameters.log.json"
        log_file.touch()

        logger.info(f"Saving log file in {log_file}")
        with open(log_file, "w") as log:
            json.dump(self.dict(), log, indent=2)

        if self.log_weights:
            weights_dir = exp_folder / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
        else:
            weights_dir = None

        if self.lrmin > self.lrmax:
            raise ValueError(f"lrmin ({self.lrmin}) should be smaller than or equal to lmax ({self.lrmax})")

        # Defining the grid of hyperparameters
        lr_tab = np.exp(np.linspace(np.log(self.lrmin), np.log(self.lrmax), self.num_exp_lr))
        bs_tab = np.linspace(self.bs_min, self.bs_max, self.num_exp_bs, dtype=np.int64)

        experiment_results = {}

        logger.info(f"Launching {self.num_exp_lr * self.num_exp_bs} experiences")

        # WANDB group
        group = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").split(".")[0]
        exp_num = 0
        experiment_results = {}

        for k in range(min(self.num_exp_lr, len(lr_tab))):
            for j in range(min(self.num_exp_bs, len(bs_tab))):

                # Initial weights should be stored in

                if self.log_weights:
                    save_weights_file = weights_dir / f"weights_{exp_num}.pth"
                else:
                    save_weights_file = None

                # Uncomment for wandb logging
                # reset_wandb_env()
                # wandb.init(project=self.project_name,
                        #    config=self.dict(),
                        #    group=group)

                # Here the seed is not changed
                logger.info(f"EXPERIENCE NUMBER {k}:{j}")

                exp_dict = risk_analysis(
                    self.iterations,
                    int(bs_tab[j]),
                    self.batch_size_eval,
                    lr_tab[k],
                    self.eval_freq,
                    self.dataset,
                    self.data_path,
                    self.model,
                    str(exp_folder),
                    self.depth,
                    self.width,
                    self.optim,
                    self.min_points,
                    self.seed,
                    save_weights_file,
                    self.compute_dimensions,
                    self.initial_weights,
                    ripser_points=self.ripser_points,
                    jump=self.jump,
                    additional_dimensions=self.additional_dimensions,
                    data_proportion=self.data_proportion
                )

                experiment_results[exp_num] = exp_dict

                save_path = Path(exp_folder) / f"results_{exp_num}.json"
                with open(str(save_path), "w") as save_file:
                    json.dump(experiment_results, save_file, indent=2)

                # Remove previously saved file
                if exp_num >= 1:
                    if (Path(exp_folder) / f"results_{exp_num - 1}.json").exists():
                        os.remove(Path(exp_folder) / f"results_{exp_num - 1}.json")

                # Uncomment for Wandb logging
                # wandb.join()
                # wandb.log(exp_dict)

                exp_num += 1

        return str(exp_folder)


if __name__ == "__main__":
    fire.Fire(AnalysisOptions)
