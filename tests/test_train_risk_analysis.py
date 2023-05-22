import json
import tempfile
from pathlib import Path

from models.fc import fc_mnist
from PHDim.__main__ import AnalysisOptions


HYPERPARAMETERS = ["iterations",
                   "batch_size_eval",
                   "lrmin",
                   "lrmax",
                   "eval_freq",
                   "dataset",
                   "data_path",
                   "model",
                   "save_folder",
                   "depth",
                   "width",
                   "optim",
                   "data_proportion",
                   "min_points",
                   "num_exp_lr",
                   "num_exp_bs"
                   ]

KEYS = [
    "ph_dim_euclidean",
    "ph_dim_hybrid_based",
    "ph_dim_losses_based",
    "train_acc",
    "eval_acc",
    "acc_gap",
    "loss_gap",
    "test_loss",
    "learning_rate",
    "batch_size",
    "LB_ratio",
    "depth",
    "width",
    "model"
]


class FileNotCreatedError(BaseException):
    ...


class WeightsUncorrectlySaved(BaseException):
    ...


def test_train_risk_analysis():

    with tempfile.TemporaryDirectory() as output_dir:

        options = AnalysisOptions(iterations=20,
                                  data_proportion=0.001,
                                  dataset="mnist",
                                  save_folder=output_dir,
                                  min_points=1,
                                  num_exp_bs=1,
                                  num_exp_lr=2,
                                  depth=2,
                                  width=10,
                                  bs_min=2,
                                  bs_max=4,
                                  batch_size_eval=2,
                                  log_weights=True,
                                  compute_dimensions=False,
                                  project_name="pytest",
                                  ripser_points=10,
                                  jump=5,
                                  resize=8)

        exp_folder = options()

        exp_folder = Path(exp_folder)
        assert exp_folder.is_dir(), str(exp_folder)

        log_file = exp_folder / "parameters.log.json"
        result_file = exp_folder / "results_1.json"

        # Test hyperparameters logging
        if not log_file.exists():
            raise FileNotCreatedError(f"File {str(log_file)} not created!")

        with open(str(log_file), "r") as logs:
            log_dict = json.load(logs)
            assert all([k in log_dict.keys() for k in HYPERPARAMETERS]), log_dict.keys()

        # Test results logging
        if not result_file.exists():
            raise FileNotCreatedError(f"File {str(result_file)} not created!")

        with open(str(result_file), "r") as results:
            result_dict = json.load(results)

            assert len(result_dict.keys()) == 2, len(result_dict.keys())

            for exp_num in [0, 1]:
                assert str(exp_num) in result_dict.keys(), result_dict.keys()

        # Testing that weights have been saved
        weights_dir = exp_folder / "weights"
        assert weights_dir.is_dir(), str(weights_dir)

        for exp_num in [0, 1]:
            if not (weights_dir / f"weights_{exp_num}.pth").exists():
                raise FileNotCreatedError(f"Weights file weights_{exp_num}.pth not created")
