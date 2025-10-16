# %%
import torch

WEIGHT_DECAY = 0

# hyperparameter tuning
ntrials = 100

run_config = {
    "num_epochs": 150,
    "repeats": 1024,
    "val_split_percentage": 0.04,
    "test_split_percentage": 0.04,
}

default_model_hyperparameters = {
    "linDropoutRate": 0.3781699841848126,
    "convDropoutRate": 0.008987655766637417,
}

default_training_hyperparameters = {
    "optimizer": torch.optim.SGD,
    "lr": 0.009402148211490864
}

# logs
label = "Test set tracking"
modification = "Modified the code, to also track the test set for the first time."
