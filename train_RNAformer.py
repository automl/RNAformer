from typing import List
import os, sys, socket
import argparse, collections, yaml
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import logging
import torch.cuda
import pytorch_lightning as pl
import numpy as np

from RNAformer.pl_module.datamodule_rna import DataModuleRNA
from RNAformer.pl_module.rna_folding_trainer import RNAFoldingTrainer

from RNAformer.utils.configuration import Config
from RNAformer.utils.instantiate import instantiate
from RNAformer.utils.folder_manager import get_experiment_folder


def bold(msg):
    return f"\033[1m{msg}\033[0m"


def main(cfg):
    """
    Launch pretraining
    """

    if os.environ.get("LOCAL_RANK") is None or os.environ.get("LOCAL_RANK") == 0:
        is_rank_zero = True
        rank = 0
    else:
        is_rank_zero = False
        rank = os.environ.get("LOCAL_RANK")

    if cfg.resume_training:
        exp_folder = get_experiment_folder(**cfg.experiment, new_folder=False)
    else:
        exp_folder = get_experiment_folder(**cfg.experiment, new_folder=is_rank_zero)

    if isinstance(cfg.trainer.devices, str):
        cfg.trainer.devices = list(map(int, cfg.trainer.devices.split(",")))
        cfg.rna_data.num_gpu_worker = len(cfg.trainer.devices)

    logger = logging.getLogger(__name__)

    if is_rank_zero:
        cfg.save_config(exp_folder)

        logging.basicConfig(
            format="[%(asctime)s][%(levelname)s][%(name)s] - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(exp_folder / "logfile.txt")],
        )

        logger.info(bold("######################################################"))
        logger.info(bold("########          START   TRAINING          ##########"))
        logger.info(bold("######################################################"))

        logger.info(f"########  Project:    {cfg.experiment.project_name}")
        logger.info(f"########  Session:    {cfg.experiment.session_name}")
        logger.info(f"########  Experiment: {cfg.experiment.experiment_name}")
        logger.info(f"save logs and checkpoints in: {exp_folder.as_posix()}")

        logger.info(bold("############### CONFIGURATION"))
        logger.info("RNA Task args")
        logger.info(cfg.rna_data)
        logger.info("Trainer args")
        logger.info(cfg.trainer)
        logger.info("Train args")
        logger.info(cfg.train)
        logger.info("Deepspeed args")
        logger.info(cfg.deepspeed)
        logger.info("Optimizer args")
        logger.info(cfg.train.optimizer)
        logger.info("RNAformer args")
        logger.info(cfg.RNAformer)

    # Set seed before initializing model
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    torch.cuda.manual_seed_all(cfg.train.seed)

    logger.info(bold(f"############### LOAD DATA on rank {rank}"))

    data_module = DataModuleRNA(**cfg.rna_data, logger=logger)

    if cfg.rna_data.max_len > 200:
        cfg.RNAformer.max_len = 500  # cfg.rna_data.max_len
    else:
        cfg.RNAformer.max_len = 200

    cfg.RNAformer.seq_vocab_size = data_module.seq_vocab_size
    cfg.RNAformer.trg_vocab_size = data_module.struct_vocab_size

    model_module = RNAFoldingTrainer(
        cfg_train=cfg.train,
        cfg_model=cfg.RNAformer,
        py_logger=logger,
        val_sets_name=data_module.valid_sets,
        ignore_index=data_module.ignore_index,
    )

    if is_rank_zero:
        def count_parameters(parameters):
            return sum(p.numel() for p in parameters if p.requires_grad)

        logger.info(f"#### trainable_parameters {count_parameters(model_module.parameters())}")

        def print_model_param_stats(model):
            for idx, (name, params) in enumerate(model.named_parameters()):
                logger.info(
                    f"{idx:03d} {name:70} shape:{str(list(params.shape)):12} mean:{params.mean():8.4f} std:{params.std():8.6f} grad: {params.requires_grad}")

        print_model_param_stats(model_module.model)

    if cfg.resume_training:
        logger.info(bold(f"############### RESUME TRAINING on rank {rank}"))

    logger.info(f'#### Load logger on rank {rank}')
    training_logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir=exp_folder,
        name="",
        version="tb",
        prefix="",
    )

    logger.info(f"#### Load callbacks on rank {rank}")
    callbacks: List[pl.Callback] = []
    if "callbacks" in config:
        for cb_name, cb_conf in config.callbacks.items():
            if cb_conf is not None and "_target_" in cb_conf:
                logger.info(f"Instantiating callback <{cb_name}>")
                if "dirpath" in cb_conf:
                    cb_conf["dirpath"] = exp_folder
                callbacks.append(instantiate(cb_conf))

    logger.info(f'#### Load strategy on rank {rank}')
    if cfg.trainer.devices == 1:
        strategy = pl.strategies.DDPStrategy(
            find_unused_parameters=True,
            static_graph=True
        )
    else:
        strategy = pl.strategies.DeepSpeedStrategy(
            **cfg.deepspeed,
            remote_device=None,  # Initialize directly on GPUs instead of CPU (ZeRO-3)
        )

    #  checkout https://pytorch-lightning.readthedocs.io/en/stable/extensions/plugins.html
    plugins = []

    logger.info(bold(f"############### TRAINER on rank {rank}"))
    cfg.trainer.num_nodes = 1  # uses multiple GPUs but all on 1 instance

    trainer = instantiate(cfg.trainer, instance=pl.Trainer,
                          callbacks=callbacks,
                          plugins=plugins,
                          strategy=strategy,
                          logger=training_logger,
                          )

    logger.info(f"Starting training on rank {rank}")
    trainer.fit(
        model=model_module, datamodule=data_module, ckpt_path=cfg.resume_training
    )

    if is_rank_zero:
        logger.info(f"Saving model to {exp_folder} on rank {rank}")
        trainer.save_checkpoint(exp_folder / "final_weights.ckpt", weights_only=True)
        logger.info(f"Finished saving model weights on rank {rank}")
    # Barrier avoids checkpoint corruption if node 0 exits earlier than other
    # nodes triggering termination of other nodes
    logger.info(f"Wait on barrier: rank {rank}")
    torch.distributed.barrier()

    logger.info("End training!")


if __name__ == "__main__":

    from functools import reduce  # forward compatibility for Python 3
    import operator


    def update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d


    def getFromDict(dataDict, mapList):
        return reduce(operator.getitem, mapList, dataDict)


    def setInDict(dataDict, mapList, value):
        getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


    def convert_string_value(value):
        if value in ('false', 'False'):
            value = False
        elif value in ('true', 'True'):
            value = True
        else:
            try:
                value = int(value)
            except:
                try:
                    value = float(value)
                except:
                    pass
        return value


    default_config_name = "default_config.yaml"

    parser = argparse.ArgumentParser(description='Train RNAformer')
    parser.add_argument('-c', '--config', type=str, default=default_config_name, help='config file name')

    args, unknown_args = parser.parse_known_args()

    config_name = args.config
    if not config_name.endswith('.yaml'):
        config_name += '.yaml'

    config_file = os.path.join("config", args.config)
    with open(config_file, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)

    for arg in unknown_args:
        if '=' in arg:
            keys = arg.split('=')[0].split('.')
            value = convert_string_value(arg.split('=')[1])
            print(keys, value)
            setInDict(config_dict, keys, value)
        else:
            raise UserWarning(f"argument unknown: {arg}")

    config = Config(config_dict=config_dict)

    main(cfg=config)
