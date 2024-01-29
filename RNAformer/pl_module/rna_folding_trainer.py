from collections import defaultdict, Counter
import pytorch_lightning as pl
import torch
import inspect

from RNAformer.model.RNAformer import RiboFormer
from RNAformer.utils import instantiate
from RNAformer.utils.group_parameters import group_parameters_for_optimizer
from RNAformer.utils.optim.lr_schedule import get_learning_rate_schedule


class RNAFoldingTrainer(pl.LightningModule):

    def __init__(
            self,
            cfg_train,
            cfg_model,
            py_logger,
            data_module,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.cfg_train = cfg_train
        self.data_module = data_module

        self.val_sets_name = data_module.valid_sets
        self.ignore_index = data_module.ignore_index
        self.py_logger = py_logger

        self.model = RiboFormer(cfg_model)

        if not hasattr(cfg_model, "binary_output") or cfg_model.binary_output:
            self.loss_train = torch.nn.BCEWithLogitsLoss(reduction='none')
            self.loss_valid = torch.nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_train = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none',
                                                        label_smoothing=0.0)
            self.loss_valid = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none',
                                                        label_smoothing=0.0)

        self.intern_log = []
        self.validation_step_outputs = defaultdict(list)

    def on_train_start(self):

        if self.cfg_train.optimizer.scheduler_mult_factor is not None:
            self.py_logger.info(
                f"Multiplying all LR schedule lambas by {self.cfg_train.optimizer.scheduler_mult_factor}"
            )
            self.lr_schedulers().lr_lambdas = [
                lambda x: self.cfg_train.optimizer.scheduler_mult_factor * fn(x)
                for fn in self.lr_schedulers().lr_lambdas
            ]

    def training_step(self, batch, batch_idx):

        if self.model.cycling:
            max_cycle = min(max(1, self.current_epoch // 3), self.model.cycle_steps)
        else:
            max_cycle = 0

        logits_mat, mask = self.model(batch['src_seq'], batch['length'], batch['pdb_sample'], max_cycle=max_cycle)

        pred = logits_mat[batch['mask']][:, 0]
        target = batch['trg_mat'][batch['mask']].float()
        loss = self.loss_train(pred, target)
        loss = loss[~torch.isnan(loss)]
        loss = torch.mean(loss)

        self.log(
            f"train/loss",
            loss.detach(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )

        self.log("global_step", torch.FloatTensor([self.global_step]))

        return {"loss": loss,
                "mean_batch_length": torch.mean(batch['length'].to(torch.float)),
                "batch_size": batch['length'].shape[0],
                "count": batch['length'].shape[0]}

    def on_train_epoch_start(self) -> None:
        self.data_module.train_seed = self.global_step + self.cfg_train.seed

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        metrics = defaultdict(list)

        with torch.no_grad():
            logits, mask = self.model(batch['src_seq'], batch['length'], batch['pdb_sample'])

            pred = logits[batch['mask']][:, 0]
            target = batch['trg_mat'][batch['mask']].float()
            loss = self.loss_valid(pred, target)
            loss = loss[~torch.isnan(loss)]
            loss = torch.mean(loss)

            for b, length in enumerate(batch['length'].detach().cpu().numpy()):
                true_mat = batch['trg_mat'][b, :length, :length].to(torch.bool)
                pred_mat = torch.sigmoid(logits[b, :length, :length, -1]) > 0.5

                solved = torch.equal(true_mat, pred_mat).__int__()
                metrics['solved'].append(
                    torch.tensor([solved], dtype=true_mat.dtype, device=true_mat.device, requires_grad=False))

                tp = torch.logical_and(pred_mat, true_mat).sum()
                tn = torch.logical_and(torch.logical_not(pred_mat), torch.logical_not(true_mat)).sum()
                fp = torch.logical_and(pred_mat, torch.logical_not(true_mat)).sum()
                fn = torch.logical_and(torch.logical_not(pred_mat), true_mat).sum()
                assert pred_mat.size().numel() == tp + tn + fp + fn
                accuracy = tp / pred_mat.size().numel()
                precision = tp / (1e-4 + tp + fp)
                recall = tp / (1e-4 + tp + fn)
                f1_score = 2 * tp / (1e-4 + (2 * tp + fp + fn))
                mcc = (tp * tn - fp * fn) / (1e-4 + torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
                metrics['accuracy'].append(accuracy)
                metrics['precision'].append(precision)
                metrics['recall'].append(recall)
                metrics['f1_score'].append(f1_score)
                metrics['mcc'].append(mcc)

            self.log(
                f"val/{self.val_sets_name[dataloader_idx]}/loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=dataloader_idx == None or dataloader_idx == 0,
                sync_dist=True,
            )

        return_dict = {"loss": loss,
                       "batch_length": torch.mean(batch['length'].float()),
                       "count": batch['length'].shape[0]}

        metrics = {k: torch.stack(v).sum() for k, v in metrics.items()}
        return_dict.update(metrics)

        self.validation_step_outputs[dataloader_idx].append(return_dict)

        return return_dict

    def on_validation_epoch_end(self):

        values = ["f1_score", "solved", "batch_length", "count"]

        summed_values_all = Counter()

        for dataset_idx, dataset_name in enumerate(self.val_sets_name):

            output = self.validation_step_outputs[dataset_idx]
            if len(output) < 1:
                continue

            summed_values = Counter()
            for out_dict in output:
                for key in values:
                    summed_values[key] += out_dict[key]
                    if "ts" in dataset_name:
                        summed_values_all[key] += out_dict[key]

            metrics = {}
            for k in values:
                if k not in ["batch_length", "count"]:
                    metrics[k] = summed_values[k] / summed_values['count']

            for name, value in metrics.items():
                self.log(f"val/{dataset_name}/{name}", value,
                         on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, )
                if self.local_rank == 0:
                    print(f"val/{dataset_name}/{name}", value, self.local_rank)

        metrics = {}
        for k in values:
            if k not in ["batch_length", "count"]:
                metrics[k] = summed_values_all[k] / summed_values_all['count']

        for name, value in metrics.items():
            self.log(f"val/all_ts/{name}", value,
                     on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, )
            if self.local_rank == 0:
                print(f"val/all_ts/{name}", value, self.local_rank)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):

        parameters = group_parameters_for_optimizer(self.model, self.cfg_train.optimizer,
                                                    normalization_regularization=self.cfg_train.optimizer_param_grouping.normalization_regularization,
                                                    bias_regularization=self.cfg_train.optimizer_param_grouping.bias_regularization, )
        optimizer = instantiate(self.cfg_train.optimizer, parameters)

        for i, g in enumerate(optimizer.param_groups):
            ntensors = len(g['params'])
            nparams = sum(p.numel() for p in g['params'])
            hparams = {k: v for k, v in g.items() if k != 'params'}
            self.py_logger.info(f'Optimizer group {i}: {ntensors} tensors, {nparams} parameters, {hparams}')

        if 'scheduler' not in self.cfg_train:
            return optimizer
        else:
            lr_lambda = get_learning_rate_schedule(self.cfg_train.scheduler)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

            return [optimizer], {'scheduler': lr_scheduler,
                                 'interval': 'step',
                                 'monitor': self.cfg_train.get('scheduler_monitor', 'val/loss')}

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        # https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html#set-grads-to-none
        # TD [2022-04-30]: DeepSpeed optimizer uses the kwarg set_grad_to_none instead of set_to_none
        if 'set_to_none' in inspect.signature(optimizer.zero_grad).parameters:
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad()
