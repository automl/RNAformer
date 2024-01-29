#!/bin/bash

mkdir -p datasets

wget -P datasets https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/biophysical_model_data.plk
wget -P datasets https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/bprna_data.plk
wget -P datasets https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/experimental_pretrain_data.plk
wget -P datasets https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/intra_family_experimental_data.plk
wget -P datasets https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/inter_family_experimental_data.plk
wget -P datasets https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/datasets/test_sets.plk
