import os
import pathlib


def get_experiment_folder(experiments_base_dir, project_name, session_name, experiment_name, new_folder=False,
                          count_folder=True):
    exp_dir = pathlib.Path(experiments_base_dir) / project_name / session_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    counter = 0
    if count_folder:
        counting_name = experiment_name + f"-{counter:03d}"
        while os.path.isfile(exp_dir / counting_name) or os.path.isdir(exp_dir / counting_name):
            counter += 1
            counting_name = experiment_name + f"-{counter:03d}"
            if not new_folder:
                counting_name = experiment_name + f"-{max(0, counter - 1):03d}"
    else:
        counting_name = experiment_name

    experiment_folder = exp_dir / counting_name
    experiment_folder.mkdir(parents=True, exist_ok=True)
    return experiment_folder
