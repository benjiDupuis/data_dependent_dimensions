import torch

from topology import calculate_ph_dim, fast_ripser

SIZE = 1000


def test_calculate_ph_dim():

    w = torch.rand(SIZE, 10)
    dimension = calculate_ph_dim(w)
    assert dimension, dimension


def test_fast_ripser():

    w = torch.rand(SIZE, 10)
    dimension = fast_ripser(w)
    assert dimension, dimension
