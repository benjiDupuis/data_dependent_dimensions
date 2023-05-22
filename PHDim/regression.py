import datetime
import json
import os
from collections import deque
from pathlib import Path

import fire
import numpy as np
import torch
import wandb
from loguru import logger
from pydantic import BaseModel
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from analysis.kendall import granulated_kendall_from_json
from models.fc import fc_bhp, AttentionFCNN
from topology import fast_ripser


DATA_SEED = 56  # For splitting in test:train


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def get_weights(net) -> torch.Tensor:
    with torch.no_grad():
        w = []

        # TODO: improve this?
        for p in net.parameters():
            w.append(p.view(-1).detach().to(torch.device('cpu')))
        return torch.cat(w)


@torch.no_grad()
def eval_bhp(train_x, train_y, net, obj) -> float:

    net.eval()
    estimated_y = net(train_x)
    assert estimated_y.ndim == 2
    assert estimated_y.shape[1] == 1
    losses = (estimated_y - train_y).pow(2)
    loss = losses.mean().cpu().item()
    return loss, losses.flatten() / train_x.shape[0]


@torch.no_grad()
def eval_non_lipschitz(train_x, train_y, net, obj) -> float:

    net.eval()
    estimated_y = net(train_x)
    assert estimated_y.ndim == 2
    assert estimated_y.shape[1] == 1
    losses = train_x.pow(2).sum(1) * (estimated_y - train_y).pow(2)
    loss = losses.mean().cpu().item()
    return loss, losses.flatten()


VALIDATION_PROPORTION = 0.2
STOPPING_CRITERION = 0.005


class UnknownDatasetError(BaseException):
    ...


class UnknownModelError(BaseException):
    ...


def train_one_model(eval_freq: int = 1000,
                    lr: float = 1.e-3,
                    iterations: int = 100000,
                    width: int = 1000,
                    depth: int = 7,
                    batch_size: int = 32,
                    ripser_points: int = 3000,
                    jump: int = 20,
                    min_points: int = 200,
                    dataset: str = "california",
                    model: str = "fcnn",
                    stopping_criterion: float = STOPPING_CRITERION,
                    ph_period: int = None,
                    additional_dimensions: bool = False):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # data
    if dataset == "boston":
        dataset, targets = load_boston(return_X_y=True)
    elif dataset == "california":
        dataset, targets = fetch_california_housing(return_X_y=True)
    else:
        raise UnknownDatasetError(f"Dataset {dataset} is unknown or not supported")
    dataset = dataset.astype(np.float64)
    targets = targets.astype(np.float64)
    dataset = (dataset - dataset.mean(0)) / dataset.std(0)
    # dataset = (dataset - dataset.mean(0))

    training_set, test_set, training_targets, test_targets = train_test_split(dataset,
                                                                              targets,
                                                                              test_size=VALIDATION_PROPORTION,
                                                                              random_state=DATA_SEED)

    np.random.seed(DATA_SEED)

    # Turning data into tensors
    training_set = torch.from_numpy(training_set.astype(np.float32)).to(device)
    test_set = torch.from_numpy(test_set.astype(np.float32)).to(device)
    training_targets = torch.from_numpy(training_targets.reshape(-1, 1).astype(np.float32)).to(device)
    test_targets = torch.from_numpy(test_targets.reshape(-1, 1).astype(np.float32)).to(device)

    # Defining dataloaders
    dataset_train = torch.utils.data.TensorDataset(training_set, training_targets)
    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    def cycle_training(loader):
        while True:
            for data in loader:
                yield data
    cycle_dataloader = cycle_training(dataloader)

    # model
    input_dim = training_set.shape[1]

    if model == "fcnn":
        net = fc_bhp(width=width, depth=depth, input_dim=input_dim).to(device)
    elif model == "attention":
        net = AttentionFCNN(depth=depth, width=width, input_dim=input_dim).to(device)
    else:
        raise UnknownModelError(f"Model {model} not implemented")
    print(net)
    n_weights = 0
    for p in net.parameters():
        n_weights += p.flatten().shape[0]

    # Defining objectives and optimizers
    obj = torch.nn.MSELoss()
    optim = torch.optim.SGD(net.parameters(), lr=lr)

    STOP = False
    risk_hist, eval_hist, batch_risk_hist = [], [], []

    weights_history = deque([])
    batch_risk_history = deque([])
    outputs_history = deque([])
    eval_history = deque([])

    # previous_weights = None
    previous_train_loss = None
    CONVERGED = False
    exp_results = {}
    avg_train_loss = 0.

    logger.info("Starting training")
    for i, (x, y) in enumerate(cycle_dataloader):

        if i % eval_freq == 0:
            loss_eval, _ = eval_bhp(test_set, test_targets, net, obj)
            eval_hist.append(loss_eval)
            logger.info(f"Evaluation at iteration {i} finished ✅, score (deviation): {np.sqrt(loss_eval)}")

            loss_train, features = eval_bhp(training_set, training_targets, net, obj)
            logger.info(f"Evaluation on training set at iteration {i} finished ✅, score (deviation): {np.sqrt(loss_train)}")

            avg_train_loss += loss_train
            wandb.log({"training set loss": loss_train})

            # Stopping criterion on instant train loss
            if (i > 0) and (np.abs(loss_train - previous_train_loss) / previous_train_loss < stopping_criterion):
                if not CONVERGED:
                    logger.info(f"Experiment converged in {i} iterations !!")
                    CONVERGED = True

            previous_train_loss = loss_train

        net.train()

        x, y = x.to(device), y.to(device)

        optim.zero_grad()
        out = net(x)
        loss = obj(out, y)

        if torch.isnan(loss):
            logger.error('Loss has gone nan ❌')
            break

        # calculate the gradients
        loss.backward()

        # take the step
        optim.step()

        # Some logging
        batch_risk_history.append([loss.cpu().item()])
        batch_risk_hist.append([i, loss.cpu().item()])

        if i > iterations:
            CONVERGED = True

        if CONVERGED or ((ph_period is not None) and (not CONVERGED) and (i % ph_period == 0)):
            weights_history.append(get_weights(net))
            loss_train, features = eval_bhp(training_set, training_targets, net, obj)
            outputs_history.append(features)
            loss_eval, features = eval_bhp(test_set, test_targets, net, obj)
            eval_history.append(features)

        if len(weights_history) >= ripser_points:
            STOP = True
            if ph_period is not None:
                weights_history.popleft()
                outputs_history.popleft()
                eval_history.popleft()

                # clear cache
        torch.cuda.empty_cache()

        if STOP and CONVERGED:

            if len(weights_history) < ripser_points:
                logger.warning("Experiment did not converge")
                break

            loss_eval, _ = eval_bhp(test_set, test_targets, net, obj)
            eval_hist.append(loss_eval)

            loss_train, _ = eval_bhp(training_set, training_targets, net, obj)
            risk_hist.append([i, loss_train])

            logger.info(f"Final sqrt(losses): train: {round(np.sqrt(loss_train), 2)}, eval: {round(np.sqrt(loss_eval), 2)}")

            weights_history_np = torch.stack(tuple(weights_history)).cpu().numpy()
            outputs_history_np = torch.stack(tuple(outputs_history)).cpu().numpy()
            eval_history_np = torch.stack(tuple(eval_history)).cpu().numpy()

            del weights_history

            jump_size = int((ripser_points - min_points) / jump)

            logger.info("Computing euclidean PH dim...")
            ph_dim_euclidean = fast_ripser(weights_history_np,
                                           max_points=ripser_points,
                                           min_points=min_points,
                                           point_jump=jump_size)

            logger.info("Computing PH dim in output space...")
            ph_dim_losses_based = fast_ripser(outputs_history_np,
                                              max_points=ripser_points,
                                              min_points=min_points,
                                              point_jump=jump_size,
                                              metric="manhattan")

            logger.debug(f"outputs shape: {outputs_history_np.shape}")

            logger.info("Computing PH dim in eval space...")
            ph_dim_eval_based = fast_ripser(eval_history_np,
                                            max_points=ripser_points,
                                            min_points=min_points,
                                            point_jump=jump_size)

            subset_dim_dict = {}
            if additional_dimensions:
                PERCENTAGES = [2, 10, 20, 30, 40, 50, 60, 70, 80, 90]

                for ph_perc in PERCENTAGES:
                    n_ph = int(outputs_history_np.shape[1] * ph_perc / 100.)
                    logger.debug(f"n_ph: {n_ph}")
                    ph_dim_subset_based = fast_ripser(outputs_history_np[..., :n_ph],
                                                      max_points=ripser_points,
                                                      min_points=min_points,
                                                      point_jump=jump_size,
                                                      metric="manhattan")
                    subset_dim_dict.update({
                        f"ph_dim_subset_based_{ph_perc}": ph_dim_subset_based
                    })

            exp_results = {
                "ph_dim_euclidean": ph_dim_euclidean,
                "ph_dim_losses_based": ph_dim_losses_based,
                "ph_dim_eval_based": ph_dim_eval_based,
                "train_loss": loss_train,
                "eval_loss": loss_eval,
                "loss_gap": np.abs(loss_train - loss_eval),
                "learning_rate": lr,
                "batch_size": int(batch_size),
                "LB_ratio": lr / batch_size
            }
            exp_results.update(subset_dim_dict)

            break

    return exp_results


class BHPAnalysis(BaseModel):

    eval_freq: int = 2000
    output_dir: str = "./bhp_experiments"
    iterations: int = 10000000
    seed: int = 1234
    save_outputs: bool = False
    project_name: str = "generalization_ph_BHP"
    width: int = 100
    depth: int = 5
    ripser_points: int = 8000
    jump: int = 20
    min_points: int = 2000
    dataset: str = "california"
    model: str = "fcnn"
    bs_min: int = 32
    bs_max: int = 200
    lr_min: float = 1.e-3
    lr_max: float = 0.015
    stopping_criterion: float = STOPPING_CRITERION
    ph_period: int = None  # period at which points are taken, if None it will be at the end
    additional_dimensions: bool = False

    def __call__(self):

        output_dir = Path(self.output_dir)

        lr_tab = np.exp(np.linspace(np.log(self.lr_min), np.log(self.lr_max), 6))
        bs_tab = np.linspace(self.bs_min, self.bs_max, 6, dtype=np.int64)

        exp_folder = output_dir / str(datetime.datetime.now()).replace(" ", "_").replace(":", "_")
        exp_folder.mkdir(parents=True, exist_ok=True)

        log_file = exp_folder / "parameters.log.json"
        log_file.touch()

        logger.info(f"Saving log file in {log_file}")
        with open(log_file, "w") as log:
            json.dump(self.dict(), log, indent=2)

        experiment_results = {}
        group = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").split(".")[0]

        num_exp = 0

        for k in range(len(lr_tab)):
            for b in range(len(bs_tab)):

                reset_wandb_env()
                wandb.init(project=self.project_name,
                           config=self.dict(),
                           group=group)

                logger.info(f"EXPERIENCE NUMBER {num_exp}")

                exp_dict = train_one_model(self.eval_freq,
                                           lr_tab[k],
                                           self.iterations,
                                           self.width,
                                           self.depth,
                                           int(bs_tab[b]),
                                           self.ripser_points,
                                           self.jump,
                                           self.min_points,
                                           self.dataset,
                                           self.model,
                                           self.stopping_criterion,
                                           self.ph_period,
                                           self.additional_dimensions)

                wandb.log(exp_dict)
                experiment_results[num_exp] = exp_dict

                save_path = Path(exp_folder) / f"results_{num_exp}.json"
                with open(str(save_path), "w") as save_file:
                    json.dump(experiment_results, save_file, indent=2)

                # Remove previously saved file
                if num_exp >= 1:
                    if (Path(exp_folder) / f"results_{num_exp - 1}.json").exists():
                        os.remove(Path(exp_folder) / f"results_{num_exp - 1}.json")

                num_exp += 1
                wandb.join()

        last_json_file = str(Path(exp_folder) / f"results_{num_exp - 1}.json")
        granulated_kendall_from_json(last_json_file, generalization_key="loss_gap")
        logger.info(f"All {num_exp} experience(s) finished correctly ✅")


if __name__ == "__main__":
    fire.Fire(BHPAnalysis)
