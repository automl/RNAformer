#!/bin/bash

python infer_riboformer_evaluate.py -y -c 6 --state_dict models/RNAformer_32M_state_dict_biophysical.pth --config models/RNAformer_32M_config_biophysical.yml

python infer_riboformer_evaluate.py -b -c 6 --state_dict models/RNAformer_32M_state_dict_bprna.pth --config models/RNAformer_32M_config_bprna.yml

python infer_riboformer_evaluate.py -c 6 --state_dict models/RNAformer_32M_state_dict_intra_family_finetuned.pth --config models/RNAformer_32M_config_intra_family_finetuned.yml

python infer_riboformer_evaluate.py -c 6 --state_dict models/RNAformer_32M_state_dict_inter_family_finetuned.pth --config models/RNAformer_32M_config_inter_family_finetuned.yml