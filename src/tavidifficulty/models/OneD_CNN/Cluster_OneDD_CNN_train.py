# %%
from tavidifficulty.models.OneD_CNN.OneDD_CNN_train import train_model, get_data_dict
from tavidifficulty.models.OneD_CNN.OneD_CNN_config import run_config
from tavidifficulty.models.OneD_CNN.OneD_CNN_training_utils import end_of_run_logging
import logging
logging.basicConfig(level=logging.WARNING)

import os
import ray
# %%
@ray.remote(num_gpus=1/6)
def ray_training_task(data, run_config):
    return train_model(data, run_config)

# @timeit
def run():
    # remove gpus if someone else is using them
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    ray.shutdown()
    ray.init()

    data = get_data_dict()

    print(f"started cluster run with following configuration: {run_config}")

    performance_metrics_over_repeats_refs = [ray_training_task.remote(data, run_config) for _ in range(run_config['repeats'])]
    performance_metrics_over_repeats = ray.get(performance_metrics_over_repeats_refs)

    print("finished training all the models")

    end_of_run_logging(performance_metrics_over_repeats, run_config)
    
if __name__ == "__main__":
    run()
# %%