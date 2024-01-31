# RNAformer

This repository contains the source code to the paper "RNAformer" and to the workshop paper 
[*Scalable Deep Learning for RNA Secondary Structure Prediction*](https://arxiv.org/abs/2307.10073) 
accepted at the 2023 ICML Workshop on Computational Biology.

### Abstract

The field of RNA secondary structure prediction has made significant progress with the adoption of deep learning techniques. 
In this work, we present the RNAformer, a lean deep learning model using axial attention and recycling in the latent space. 
We gain performance improvements by designing the architecture for modeling the adjacency matrix directly in the latent space 
and by scaling the size of the model. Our approach achieves state-of-the-art performance on the popular benchmark datasets and 
even outperforms methods that use external information. Further, we show experimentally that the RNAformer can learn a biophysical 
model of the RNA folding process.

## Reproduce results

### Clone the repository

```
git clone https://github.com/automl/RNAformer.git
cd RNAformer
```

### Install virtual environment

The Flash Attention requeires currently a Ampere, Ada, or Hopper GPU (e.g., A100, RTX 3090, RTX 4090, H100). Support for Turing GPUs (T4, RTX 2080) is coming soon. 

```
python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt
pip install flash-attn==2.3.4
pip install -e .
```



### Download datasets

```
bash download_datasets.sh
``` 


### Download pretrained models

``` 
bash download_models.sh
```
    

### Reproduce results from the paper

``` 
bash run_evaluation.sh
```


## Infer RNAformer for RNA sequence:

An example of a inference, the script outputs position indexes in the adjacency matrix that are predicted to be paired. 

``` 
python infer_RNAformer.py -c 6 -s GCCCGCAUGGUGAAAUCGGUAAACACAUCGCACUAAUGCGCCGCCUCUGGCUUGCCGGUUCAAGUCCGGCUGCGGGCACCA --state_dict models/RNAformer_32M_state_dict_intra_family_finetuned.pth --config models/RNAformer_32M_config_intra_family_finetuned.yml
``` 

## Model Checkpoints

Please find here the state dictionaries and configs for the models used in the paper: 

RNAformer 32M from the biophysical model experiment:
```
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/RNAformer_32M_state_dict_biophysical.pth
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/RNAformer_32M_config_biophysical.yml
```

RNAformer 32M from the bprna model experiment:
```
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/RNAformer_32M_state_dict_bprna.pth
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/RNAformer_32M_config_bprna.yml
```

RNAformer 32M from the intra family finetuning experiment:
```
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/RNAformer_32M_state_dict_intra_family_finetuned.pth
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/RNAformer_32M_config_intra_family_finetuned.yml
```

RNAformer 32M from the inter family finetuning experiment:
```
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/RNAformer_32M_state_dict_inter_family_finetuned.pth
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/RNAformer_32M_config_inter_family_finetuned.yml
```