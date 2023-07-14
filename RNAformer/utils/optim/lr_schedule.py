import math
import torch


def get_learning_rate_schedule(scheduler_config):
    def lr_lambda(current_step: int):
        if current_step < scheduler_config.num_warmup_steps:
            return float(current_step) / float(max(1, scheduler_config.num_warmup_steps))
        elif scheduler_config.schedule == 'linear':
            return scheduler_config.decay_factor + (1 - scheduler_config.decay_factor) * max(0.0,
                                                                                             float(
                                                                                                 scheduler_config.num_training_steps - scheduler_config.num_warmup_steps - current_step) / float(
                                                                                                 max(1,
                                                                                                     scheduler_config.num_training_steps - scheduler_config.num_warmup_steps))
                                                                                             )
        elif scheduler_config.schedule == 'cosine':
            return scheduler_config.decay_factor + (1 - scheduler_config.decay_factor) * max(0.0,
                                                                                             (1 + math.cos(math.pi * (
                                                                                                         current_step - scheduler_config.num_warmup_steps) / float(
                                                                                                 max(1,
                                                                                                     scheduler_config.num_training_steps - scheduler_config.num_warmup_steps)))) / 2
                                                                                             )
        elif scheduler_config.schedule == 'const':
            return 1.0

    return lr_lambda
