#!/bin/bash

mkdir -p models

wget -P models https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/RNAformer_32M_state_dict_biophysical.pth
wget -P models https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/RNAformer_32M_config_biophysical.yaml

wget -P models https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/RNAformer_32M_state_dict_bprna.pth
wget -P models https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/RNAformer_32M_config_bprna.yaml

wget -P models https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/RNAformer_32M_state_dict_intra_family_finetuned.pth
wget -P models https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/RNAformer_32M_config_intra_family_finetuned.yaml

wget -P models https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/RNAformer_32M_state_dict_inter_family_finetuned.pth
wget -P models https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/RNAformer_32M_config_inter_family_finetuned.yaml




