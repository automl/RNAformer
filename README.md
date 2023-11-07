# RNAformer

This repository contains the source code to the paper 
[*Scalable Deep Learning for RNA Secondary Structure Prediction*](https://arxiv.org/abs/2307.10073) 
accepted at the 2023 ICML Workshop on Computational Biology.

### Abstract

The field of RNA secondary structure prediction has made significant progress with the adoption of deep learning techniques. In this work, we present the RNAformer, a lean deep learning model using axial attention and recycling in the latent space. We gain performance improvements by designing the architecture for modeling the adjacency matrix directly in the latent space and by scaling the size of the model. Our approach achieves state-of-the-art performance on the popular TS0 benchmark dataset and even outperforms methods that use external information. Further, we show experimentally that the RNAformer can learn a biophysical model of the RNA folding process.

## Installation

### Clone the repository

```
git clone https://github.com/automl/RNAformer.git
cd RNAformer
```

### Install conda environment 

Please adjust the cuda toolkit version in the `environment.yml` file to fit your setup. 
```
conda env create -n rnafenv -f environment.yml
conda activate rnafenv
pip install -e .
```

## Training Datasets

Please find here the datasets for the bpRNA and Rfam experiment: 

```
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/bprna_dataset.plk
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/rnafold_rfam_dataset.plk
```

## Model Checkpoints

Please find here the state dictionaries and configs for the models used in the paper: 

RNAformer 32M + cycling:
```
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/ts0_conform_dim256_cycling_32bit/config.yml
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/ts0_conform_dim256_cycling_32bit/state_dict.pth
```

RNAformer 32M:
```
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/ts0_conform_dim256_32bit/config.yml
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/ts0_conform_dim256_32bit/state_dict.pth
```

RNAformer 8M:
```
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/ts0_conform_dim128_32bit/config.yml
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/ts0_conform_dim128_32bit/state_dict.pth
```



## Inference and Evaluation

Downloads, runs and evaluates one of the 32M models (ts0_conform_dim256_32bit) from the paper (we trained three with different random seed but the performance does not differ much (std 0.002)).

```
python infer_riboformer.py -n ts0_conform_dim256_32bit  # downloads, runs and evaluates the 32M model from the paper
```
