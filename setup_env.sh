#!/bin/bash

python -m venv rnaf

source rnaf/bin/activate

pip install -U --no-cache-dir -r requirements.txt
pip install -e .

read -p "Do you want to install flash-attention from the git repository? Recommended for Nvidia Ampere, Ada, or Hopper GPUs. (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    git clone https://github.com/HazyResearch/flash-attention \
        && printf "[safe]\n\tdirectory = /flash-attention" > ~/.gitconfig \
        && git config --global --add safe.directory /home/user/flash-attention \
        && cd flash-attention && git checkout v2.3.4 \
        && cd csrc/fused_softmax && pip install . && cd ../../ \
        && cd csrc/rotary && pip install . && cd ../../ \
        && cd csrc/xentropy && pip install . && cd ../../ \
        && cd csrc/layer_norm && pip install . && cd ../../ \
        && cd csrc/fused_dense_lib && pip install . && cd ../../ \
        && cd csrc/ft_attention && pip install . && cd ../../ \
        && cd .. && rm -rf flash-attention
fi
