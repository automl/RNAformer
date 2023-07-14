
python3 train_probtransformer.py --config=default_config.yaml experiment.session_name=rna_folding_1 \
experiment.experiment_name=ts0_conform_dim64_32bit \
rna_data.dataframe_path=/home/ubuntu/tmp/data/spotrna1_conform_with_TS0.plk \
trainer.precision=32 \
RNAformer.precision=32 \
trainer.devices=8 \
trainer.max_steps=1000 \
trainer.val_check_interval=1000 \
rna_data.random_ignore_mat=0.5 \
train.seed=1234 \
train.optimizer.lr=0.001 \
train.scheduler.schedule=cosine \
rna_data.batch_token_size=500 \
rna_data.batch_size=1 \
rna_data.batch_by_token_size=true \
rna_data.oversample_pdb=1 \
rna_data.min_len=10 \
rna_data.max_len=490 \
RNAformer.model_dim=64 \
RNAformer.n_layers=6 \
RNAformer.num_head=4 \
RNAformer.cycling=False