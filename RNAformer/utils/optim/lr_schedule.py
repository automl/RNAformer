import math


def get_learning_rate_schedule(scheduler_config):
    def lr_lambda(current_step: int):

        training_steps = scheduler_config.num_training_steps - scheduler_config.num_warmup_steps

        if current_step < scheduler_config.num_warmup_steps:
            return float(current_step) / float(max(1, scheduler_config.num_warmup_steps))
        elif scheduler_config.schedule == 'linear':
            linear_decay = max(0.0, float(scheduler_config.num_training_steps - current_step) / float(
                max(1, training_steps)))
            return scheduler_config.decay_factor + (1 - scheduler_config.decay_factor) * linear_decay
        elif scheduler_config.schedule == 'cosine':
            cosine_decay = max(0.0, (1 + math.cos(
                math.pi * (current_step - scheduler_config.num_warmup_steps) / float(max(1, training_steps)))) / 2)
            return scheduler_config.decay_factor + (1 - scheduler_config.decay_factor) * cosine_decay
        elif scheduler_config.schedule == 'const':
            return 1.0

    return lr_lambda
