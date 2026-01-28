# Hyperbolic Graph Neural Networks Under the Microscope: The Role of Geometry–Task Alignment

This repository contains the code for the paper "Hyperbolic Graph Neural Networks Under the Microscope: The Role of Geometry–Task Alignment".
The general structure is based on 
- the NeurIPS 2019 paper **Hyperbolic Graph Convolutional Neural Networks**  [[Paper](https://openreview.net/forum?id=BJlSF4SgUB)] [[Code](https://github.com/HazyResearch/hgcn)] 

and its extensions
- the ACL 2022 paper **Fully Hyperbolic Neural Networks** [[Paper](https://arxiv.org/abs/2105.14686)] [[Code](https://github.com/chenweize1998/fully-hyperbolic-nn)]
- the TMLR 2025 paper **Shedding Light on Problems with Hyperbolic Graph Learning** [[Paper](https://arxiv.org/abs/2411.06688)] [[Code](https://github.com/isaykatsman/Shedding-Light-Hyperbolic-Graph-Learning)].

We unified and reimplemented many parts to fit into the more modern [PyG](https://github.com/pyg-team/pytorch_geometric) and [Geoopt](https://github.com/geoopt/geoopt) frameworks.
For the `manifolds`, we use the modules of the Geoopt library and extend it to support all manifold operations required by our models. 
This unified manifold interface allows all implementations from [Chami et al. (2019)](https://openreview.net/forum?id=BJlSF4SgUB) and [Chen et al. (2022)](https://github.com/chenweize1998/fully-hyperbolic-nn) 
to be expressed using the same underlying manifold directory, without model-specific geometric code.

# Requirements
Tested with Python 3.12.3 and PyTorch 2.5.1 (Cuda 12.4).
```
pytorch
pytorch-geometric
geoopt numpy scipy matplotlib pyyaml scikit-learn tqdm networkx pandas
```
Optional (if you want to use [wandb](https://wandb.ai/), use flag `--use_wandb` when running the code):
```
wandb
```

# How to run the code
### Reproduce experimental results
All experimental results can be reproduced using the same syntax. All use the corresponding file from `configs/`.
For example, to run 10 LP experiments using HyboNet on the Disease dataset, use:

```
python train.py --dataset disease_lp --model hybonet-lorentz --task lp
```
You can specify the usage of GPU with `--cuda <gpu_id>`, e.g., `--cuda 0`. Any negative number will use CPU.

### Compute Stress Loss after training
To compute the Stress Loss after training a model, first use the flag `--save_embeddings`, e.g.,

```
python train.py --dataset disease_lp --model hybonet-lorentz --task lp --save_embeddings
```
then use the notebook `notebooks/compute_stress_loss.iypnb` to compute the Stress Loss using the saved embeddings.
Due to storage limitations, we only provide the embeddings for the Disease dataset (LP/NC) in `data/embeddings/`.

### Sweeps
If you want to run hyperparameter sweeps, you can create a sweep configuration file in `configs/` ending with `*_sweep.yaml` together with [wandb](https://wandb.ai/).
Just add the flag `--sweep` when running the code (wandb will be turned on automatically), e.g.,

```
python train.py --dataset disease_lp --model hybonet-lorentz --task lp --sweep
```
### Hyperparameters
If you want to see a specific hyperparameter across all models and datasets, e.g. hidden dimension,
(instead of checking all configs) you can use the notebook `notebooks/hyperparameter_table.ipynb`.


### New datasets
The datasets used in [Chami et al. (2019)](https://openreview.net/forum?id=BJlSF4SgUB), [Katsman & Gilbert (2025)](https://arxiv.org/abs/2411.06688) 
and the synthetic datasets used in our paper are in `data/`.
Ours are called `pdp_tree` and `pdp_grid` for PDP task, `tree100_2_2_2_nc` and `grid_28x28_nr` for
node regression and classification.
The other datasets are from PyG datasets and are downloaded automatically when running the code for the first time.

To add a new dataset, create a new one in `data/` and add the name (if it follows our synthetic scheme) or 
method implementation to `data_utils.py` and add the corresponding configuration file in `configs/`.


### Gromov $\delta$-hyperbolicity computation
To recompute the Gromov $\delta$-hyperbolicity of all graphs in the paper, refer to `notebooks/hyperbolicity_sage_math.ipynb`.
You will need to install [SageMath](https://www.sagemath.org/) to run the notebook.
