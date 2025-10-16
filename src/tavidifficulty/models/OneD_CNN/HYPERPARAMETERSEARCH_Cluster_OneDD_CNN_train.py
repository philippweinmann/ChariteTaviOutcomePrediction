# %%
from tavidifficulty.models.OneD_CNN.OneDD_CNN_train import train_model, get_data_dict
from tavidifficulty.models.OneD_CNN.OneD_CNN_config import ntrials, run_config
from tavidifficulty.models.OneD_CNN.OneD_CNN_training_utils import evaluate_performance_metrics
import logging
import torch
logging.basicConfig(level=logging.WARNING)

import os
import optuna
import ray
# %%
@ray.remote(num_gpus=1/6)
def ray_training_task(data, run_config, model_hyperparameters, training_hyperparameters):
    return train_model(data, run_config, model_hyperparameters, training_hyperparameters)

def CV_Run(data, run_config, model_hyperparameters, training_hyperparameters):
    ray.shutdown()
    ray.init()

    performance_metrics_over_repeats_refs = [ray_training_task.remote(data, run_config, model_hyperparameters, training_hyperparameters) for _ in range(run_config['repeats'])]
    performance_metrics_over_repeats = ray.get(performance_metrics_over_repeats_refs)

    logging.info("finished training all the models")

    evaluated_performance_metrics = evaluate_performance_metrics(performance_metrics_over_repeats)

    minValLoss = min(evaluated_performance_metrics['val_losses_ageraged'])

    return minValLoss

def objective_function(trial, data):
    model_hyperparameters = {
        "linDropoutRate": trial.suggest_float('linDropoutRate', 0.3, 0.5),
        "convDropoutRate": trial.suggest_float('convDropoutRate', 0.0, 0.1),
    }

    training_hyperparameters = {
        "lr": trial.suggest_float('lr', 0.0001, 0.01, log=True),
    }

    logging.info(f"started cluster run with following configuration: {run_config}")

    minValLoss = CV_Run(data, run_config, model_hyperparameters, training_hyperparameters)
    return minValLoss

# %%
# @timeit
def run():
    # remove gpus if someone else is using them
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    data = get_data_dict()

    study = optuna.create_study(direction="minimize")
    # DO NOT USE MORE THAN ONE JOB; RAY ALREADY PARALLELIZES ENOUGH
    study.optimize(lambda trial: objective_function(trial, data=data), n_trials=ntrials, n_jobs=1)

    print(f"best params: {study.best_params}")
    print(f"best value: {study.best_value}")
    
if __name__ == "__main__":
    run()
# %%