# Data dependent fractal dimension

This is the code for our ICML 2023 paper ["Generalization Bounds using Data-Dependent Fractal Dimension"](https://arxiv.org/abs/2302.02766). 

This code builds mostly on the work of [this work.](https://arxiv.org/abs/2111.13171)

We used Wandb for logging, if you want to use it just uncomment the corresponding line (which are indicated in the script) in the script `PHDim/__main__.py`.

## installation 


You can very easily install the project with the following command:

`pip install -r requirements.txt`

All requirements are in the file `requirements.txt`.

## Pytests

To check if the installation worked, you can run pytests with the following command:

`pytest -x -vv`


## Main experiments

All scripts are run form the root. change the python path in case of import errors (`export PYTTHONPATH=$PWD` in linux). We provide simple examples, the code can easily be extended to more complex models and datasets.

### Classification

The classification experiments is in the scripts `PHDim/__main__.py` where all hyperparameters are documentated. The script can be run with the command:

`python -m PHDim [--args ARGS]` 

This basically creates a grid of hyperparameters for the specified dataset and models and run indivudual training fro each  pair of hyperparameters, those individual training are implemented in the script `¨PHDim/train_risk_analysis.py`. Training stops at convergence and then the intrinsics dimensions are computed.

The algorithm to computed those dimensions is implemented in `topology.py`, it is the same than in [this work.](https://arxiv.org/abs/2111.13171) except that the library [giotto-ph](https://pypi.org/project/giotto-ph/) is used for persistent homology, also distance matrix are precomputed only once to make the experiments faster than in the aforementioned work.

The parameter `additional_dimensions`, if set to `True` will produce the robustness experiment presented in the paper.

### Regression

Regression experiments works basically the same way except that it is implemented in a separate script, `PHDim/regression.py` for technical reasons.

### plots

The plots presented in the paper are implemented in this [script](plots/plot_results) whichj takes as inputs the **JSON** files produced by the above scripts.

## Granulated Kendall's coefficients

We implemented [Granulated Kendall's coefficients](https://arxiv.org/pdf/1912.02178.pdf) as a metric to compare different intrinsic dimensions, the codetakes as input the **JSON** files created by running an experiment and can be found in this [script](analysis/kendall.py).
  

# References

References mentioned above:

```
@article{DBLP:journals/corr/abs-2111-13171,
  author    = {Tolga Birdal and
               Aaron Lou and
               Leonidas J. Guibas and
               Umut Simsekli},
  title     = {Intrinsic Dimension, Persistent Homology and Generalization in Neural
               Networks},
  journal   = {CoRR},
  volume    = {abs/2111.13171},
  year      = {2021},
  url       = {https://arxiv.org/abs/2111.13171},
  eprinttype = {arXiv},
  eprint    = {2111.13171},
  timestamp = {Wed, 01 Dec 2021 15:16:43 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2111-13171.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org
```

```
@article{DBLP:journals/corr/abs-1912-02178,
  author    = {Yiding Jiang and
               Behnam Neyshabur and
               Hossein Mobahi and
               Dilip Krishnan and
               Samy Bengio},
  title     = {Fantastic Generalization Measures and Where to Find Them},
  journal   = {CoRR},
  volume    = {abs/1912.02178},
  year      = {2019},
  url       = {http://arxiv.org/abs/1912.02178},
  eprinttype = {arXiv},
  eprint    = {1912.02178},
  timestamp = {Thu, 02 Jan 2020 18:08:18 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1912-02178.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


