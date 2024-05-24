# RNAformer

This repository contains the source code to the preprint [*RNAformer: A Simple Yet Effective Deep Learning Model for RNA Secondary Structure Prediction*](https://www.biorxiv.org/content/10.1101/2024.02.12.579881v1)  and to the preceding workshop paper 
[*Scalable Deep Learning for RNA Secondary Structure Prediction*](https://arxiv.org/abs/2307.10073) 
presented at the 2023 ICML Workshop on Computational Biology.

### Abstract

Traditional RNA secondary structure prediction methods, based on dynamic programming, often fall short in accuracy. Recent advances in deep learning have aimed to address this, but may not adequately learn the biophysical model of RNA folding. Many deep learning approaches are also too complex, incorporating multi-model systems, ensemble strategies, or requiring external data like multiple sequence alignments. In this study, we demonstrate that a single deep learning model, relying solely on RNA sequence input, can effectively learn a biophysical model and outperform existing deep learning methods in standard benchmarks, as well as achieve comparable results to methods that utilize multi-sequence alignments. We dub this model RNAformer and achieve these benefits by a two-dimensional latent space, axial attention, and recycling in the latent space. Further, we found that our model performance improves when we scale it up. We also demonstrate how to refine a pre-trained RNAformer with fine-tuning techniques, which are particularly efficient when applied to a limited amount of high-quality data. A further aspect of our work is addressing the challenges in dataset curation in deep learning, especially regarding data homology. We tackle this through an advanced data processing pipeline that allows for training and evaluation of our model across various levels of sequence similarity. Our models and datasets are openly accessible, offering a simplified yet effective tool for RNA secondary structure prediction.

## Reproduce results

### Clone the repository

```
git clone https://github.com/automl/RNAformer.git
cd RNAformer
```

### Install virtual environment

The Flash Attention package currently requires a Ampere, Ada, or Hopper GPU (e.g., A100, RTX 3090, RTX 4090, H100). Support for Turing GPUs (T4, RTX 2080) is coming soon. 

```
python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt
pip install flash-attn==2.3.4
pip install -e .
```
Alternatively, you may install RNAformer without Flash Attention or a GPU for inference and evaluation:
```
python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### Download datasets

```
bash download_all_datasets.sh
``` 


### Download pretrained models

``` 
bash download_all_models.sh
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
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/models/RNAformer_32M_state_dict_biophysical.pth
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/models/RNAformer_32M_config_biophysical.yml
```

RNAformer 32M from the bprna model experiment:
```
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/models/RNAformer_32M_state_dict_bprna.pth
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/models/RNAformer_32M_config_bprna.yml
```

RNAformer 32M from the intra family finetuning experiment:
```
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/models/RNAformer_32M_state_dict_intra_family_finetuned.pth
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/models/RNAformer_32M_config_intra_family_finetuned.yml
```

RNAformer 32M from the inter family finetuning experiment:
```
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/models/RNAformer_32M_state_dict_inter_family_finetuned.pth
https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/models/RNAformer_32M_config_inter_family_finetuned.yml
```
