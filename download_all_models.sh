#!/bin/bash

mkdir -p models

wget -nc -P models https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/models/RNAformer_32M_state_dict_biophysical.pth
wget -nc -P models https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/models/RNAformer_32M_config_biophysical.yml

wget -nc -P models https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/models/RNAformer_32M_state_dict_bprna.pth
wget -nc -P models https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/models/RNAformer_32M_config_bprna.yml

wget -nc -P models https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/models/RNAformer_32M_state_dict_intra_family_finetuned.pth
wget -nc -P models https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/models/RNAformer_32M_config_intra_family_finetuned.yml

wget -nc -P models https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/models/RNAformer_32M_state_dict_inter_family_finetuned.pth
wget -nc -P models https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/models/RNAformer_32M_config_inter_family_finetuned.yml




