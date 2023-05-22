import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from gph.python import ripser_parallel
from sklearn.metrics.pairwise import pairwise_distances


def sample_W(W, nSamples, isRandom=True):
    n = W.shape[0]
    random_indices = np.random.choice(n, size=nSamples, replace=False)
    return W[random_indices]


def ph_dim_from_distance_matrix(dm: np.ndarray,
                                min_points=200,
                                max_points=1000,
                                point_jump=50,
                                h_dim=0,
                                alpha: float = 1.,
                                seed: int = 42) -> float:
    """
    This functions:
     - turn W into a torch tensor
     - compute the distance matrix
     - use it to compute PH dim

    :param dm: distance matrix, should be of shape (N, N)
    """
    assert dm.ndim == 2, dm
    assert dm.shape[0] == dm.shape[1], dm.shape

    np.random.seed(seed)

    test_n = range(min_points, max_points, point_jump)
    lengths = []

    for points_number in test_n:

        sample_indices = np.random.choice(dm.shape[0], points_number, replace=False)
        dist_matrix = dm[sample_indices, :][:, sample_indices]

        diagrams = ripser_parallel(dist_matrix, maxdim=0, n_threads=-1, metric="precomputed")['dgms']

        d = diagrams[h_dim]
        d = d[d[:, 1] < np.inf]
        lengths.append(np.power((d[:, 1] - d[:, 0]), alpha).sum())  # The fact that \alpha = 1 appears here

    lengths = np.array(lengths)

    # compute our ph dim by running a linear least squares
    x = np.log(np.array(list(test_n)))
    y = np.log(lengths)
    N = len(x)
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    b = y.mean() - m * x.mean()

    error = ((y - (m * x + b)) ** 2).mean()

    logger.debug(f"Ph Dimension Calculation has an approximate error of: {error}.")

    return alpha / (1 - m)


def fast_ripser(w: np.ndarray,
                min_points=200,
                max_points=1000,
                point_jump=50,
                h_dim=0,
                alpha: float = 1.,
                seed: int = 42,
                save_dir: str = None,
                metric: str = "euclidean"):

    assert w.shape[0] <= max_points, (w.shape[0], max_points)
    assert w.shape[0] >= min_points, (w.shape[0], min_points)

    starting_time = time.time()
    dm = pairwise_distances(w, metric=metric)
    logger.debug(f"Distance matrix computation time: {round(time.time() - starting_time, 2)}s")

    if save_dir is not None:
        save_path = Path(save_dir) / "distance_matrix.npy"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path), dm)

    return ph_dim_from_distance_matrix(dm,
                                       min_points,
                                       max_points,
                                       point_jump,
                                       h_dim,
                                       alpha,
                                       seed)


MAX_AUTHORIZED_DIMENSION: int = 10000


@torch.no_grad()
def distance_matrix(w: np.ndarray) -> float:
    """
    :param w: array of shape (n_vectors, dimension)
    This functions:
     - turn w into a torch tensor
     - compute the distance matrix
     - use it to compute PH dim
    """
    assert w.ndim == 2, w.shape

    # We do not use GPU to compute the distance matrix if
    # the underlying vector space dimension is greater than MAX_AUHORIZED_DIMENSION
    if w.shape[1] >= MAX_AUTHORIZED_DIMENSION:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    w = torch.from_numpy(w.astype(np.float32)).to(device) / w.shape[1]

    return torch.cdist(w, w).cpu().numpy()


def calculate_ph_dim(W: np.ndarray,
                     min_points=200,
                     max_points=1000,
                     point_jump=50,
                     h_dim=0,
                     print_error=True,
                     metric=None,
                     alpha: float = 1.,
                     seed: int = 42) -> float:
    # sample_fn should output a [num_points, dim] array

    np.random.seed(seed)

    logger.info(f"Calculating PH dimension with points {min_points} to {max_points}, seed: {seed}")

    # sample our points
    test_n = range(min_points, max_points, point_jump)
    logger.debug(f"Number of test points for PH dimension computation: {len(test_n)}")
    lengths = []
    for n in tqdm(test_n):
        if metric is None:
            # diagrams = ripser(sample_W(W, n))['dgms']
            diagrams = ripser_parallel(sample_W(W, n), maxdim=h_dim, n_threads=-1)['dgms']
        else:
            # diagrams = ripser(sample_W(W, n), metric=metric)['dgms']
            diagrams = ripser_parallel(sample_W(W, n), metric=metric, maxdim=h_dim, n_threads=-1)['dgms']

        if len(diagrams) > h_dim:
            d = diagrams[h_dim]
            d = d[d[:, 1] < np.inf]
            lengths.append(np.power((d[:, 1] - d[:, 0]), alpha).sum())  # The fact that \alpha = 1 appears here
        else:
            lengths.append(0.0)
    lengths = np.array(lengths)

    # compute our ph dim by running a linear least squares
    x = np.log(np.array(list(test_n)))
    y = np.log(lengths)
    N = len(x)
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    b = y.mean() - m * x.mean()

    error = ((y - (m * x + b)) ** 2).mean()

    if print_error:
        logger.debug(f"Ph Dimension Calculation has an approximate error of: {error}.")

    return alpha / (1 - m)


