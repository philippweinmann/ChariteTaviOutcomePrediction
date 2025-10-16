# %%
# %load_ext autoreload
# %autoreload 2
from tavidifficulty.data.datasets import CustomDataset
from tavidifficulty.preprocessing.preprocessing_extended_data import fix_columns, normalize_columns
from tavidifficulty.preprocessing.preprocess_metadata import normalize_metadata
from tavidifficulty.data.data_loading import get_extended_data
from tavidifficulty.data.data_loading import pad_if_necessary
from tavidifficulty.data.parse_results_files import parse_all_results_and_targets_files_to_patients, save_patients_to_pickle, get_patients_from_cache
from torch.utils.data import DataLoader
from tavidifficulty.models.OneD_CNN.OneD_CNN_model import BASIC_CNN1D
from tavidifficulty.models.OneD_CNN.OneD_CNN_training_utils import evaluate_performance_metrics, prepare_data_for_model_input, extract_final_accuracy_metrics, generate_plots
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from tavidifficulty.models.OneD_CNN.OneD_CNN_config import run_config, WEIGHT_DECAY, default_model_hyperparameters, default_training_hyperparameters
from functools import lru_cache
from sklearn.linear_model import LogisticRegression

import logging
logging.basicConfig(level=logging.WARNING)

# if we're using ray, it is okay to just hardcode the device.
device="cuda"

# patients = parse_all_results_and_targets_files_to_patients(use_new_targets=True, use_old_targets=False, remove_incomplete=True, remove_without_targets=True)
# %%
# torch.set_default_device(device)

# print(torch.version.cuda)
# print(torch.__version__)
# print(f"Using {device} device. Every tensor created will be by default on {device}")
# %%
# save_patients_to_pickle()
# %%
@lru_cache()
def get_padded_patients(use_old_targets, use_new_targets):
    patients = parse_all_results_and_targets_files_to_patients(use_old_targets, use_new_targets, remove_incomplete=True, remove_without_targets=True)
    # patients = get_patients_from_cache()

    input_dataframes, metadatas, y = get_extended_data(patients)

    # this is not preprocessing, this is fixing some issues, that can be directly applied to each
    # column individually, without the risk of overfitting (nested cross validation can apply later).
    # note that this has already happened for the metadatas.
    input_dataframes = fix_columns(input_dataframes)    

    y = torch.from_numpy(y)

    return input_dataframes, metadatas, y

def get_data_dict(use_old_targets = True, use_new_targets = False):
    mrt_slice_data, metadatas, y = get_padded_patients(use_old_targets=use_old_targets, use_new_targets=use_new_targets)
    max_length_without_abdominal_aorta = max(tensor.shape[0] for tensor in mrt_slice_data)

    data_dict = {
        "mrt_slice_data": mrt_slice_data,
        "metadatas": metadatas,
        "y": y,
        "max_length_without_abdominal_aorta": max_length_without_abdominal_aorta,
    }

    return data_dict


# let's randomize the val, train indexes.
def get_train_val_split_dataloaders(mrt_slice_data, metadatas, y, max_length_without_abdominal_aorta, run_config):
    # the test set has already been removed.
    X_train, X_non_train, meta_train, meta_non_train, y_train, y_non_train = train_test_split(mrt_slice_data, metadatas, y, test_size=run_config["val_split_percentage"] + run_config["test_split_percentage"])
    
    # here we apply the normalization. We do not fit on the non training data to avoid bleeding.
    X_train, X_non_train = normalize_columns(X_train, X_non_train)
    meta_train, meta_non_train = normalize_metadata(meta_train, meta_non_train)

    X_train = [prepare_data_for_model_input(input_dataframe) for input_dataframe in X_train]
    X_non_train = [prepare_data_for_model_input(input_dataframe) for input_dataframe in X_non_train]

    X_train = pad_if_necessary(padded_length=max_length_without_abdominal_aorta, X=X_train)
    X_non_train = pad_if_necessary(padded_length=max_length_without_abdominal_aorta, X=X_non_train)

    X_train = torch.from_numpy(X_train).to(device)
    X_non_train = torch.from_numpy(X_non_train).to(device)

    # let's concatenate the metadatas into one big dataframe.
    meta_train = pd.concat(meta_train)
    meta_non_train = pd.concat(meta_non_train)

    y_train = y_train.to(device)
    y_non_train = y_non_train.to(device)

    X_val, X_test, meta_val, meta_test, y_val, y_test = train_test_split(X_non_train, meta_non_train, y_non_train, test_size=run_config["test_split_percentage"] / (run_config["test_split_percentage"] + run_config["val_split_percentage"]))

    # generator=torch.Generator(device=device)
    train_dataloader = DataLoader(CustomDataset(X_train, y_train), batch_size=16, shuffle=True, drop_last=True)
    logging.info(f"training dataset: {len(train_dataloader.dataset)} patients.")
    val_dataloader = DataLoader(CustomDataset(X_val, y_val), batch_size=1, shuffle=False)
    logging.info(f"validation dataset: {len(val_dataloader.dataset)} patients.")
    # TODO return the test dataloader and evaluate it at the very end.
    test_dataloader = DataLoader(CustomDataset(X_test, y_test), batch_size=1, shuffle=False)

    metadata_y_train = copy.deepcopy(y_train).cpu()
    metadata_y_val = copy.deepcopy(y_val).cpu()
    metadata_y_test = copy.deepcopy(y_test).cpu()

    metadata_dict = {
        "meta_train": meta_train,
        "meta_train_y": metadata_y_train,
        "meta_val": meta_val,
        "meta_val_y": metadata_y_val,
        "meta_test": meta_test,
        "meta_test_y": metadata_y_test,
    }

    return train_dataloader, val_dataloader, test_dataloader, metadata_dict

def train_loop(model, train_dataloader, loss_fn, optimizer):
    total_train_loss = 0
    model.train()

    for X, y in train_dataloader:
        # Forward pass
        optimizer.zero_grad()

        outputs = model(X)
        loss = loss_fn(outputs.squeeze(), y.squeeze())
        total_train_loss += loss.item()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    logging.info(f"avg training loss: {avg_train_loss}")

    return avg_train_loss

# Test loop
def test_loop(model, dataloader, loss_fn, scheduler):
    model.eval()
    accuracy = 0
    avg_test_loss = 0
    pred_probas = []
    pred_gts = []

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            avg_test_loss += loss_fn(pred.squeeze(), y.squeeze()).item()

            pred_proba = torch.sigmoid(pred)
            pred_probas.append(pred_proba.cpu().item())
            pred_gts.append(y.cpu().item())
            pred_labels = (pred_proba >= 0.5)
            accuracy += (pred_labels.long().squeeze() == y.long()).sum().item()

    avg_test_loss /= len(dataloader)
    accuracy /= len(dataloader)

    # allows me to use the same function for the test set
    if scheduler:
        scheduler.step(avg_test_loss)

    return avg_test_loss, accuracy, pred_probas, pred_gts

def handle_metadata_log_model(model, metadata_dict):
    model.fit(metadata_dict["meta_train"], metadata_dict["meta_train_y"])

    val_preds = model.predict_proba(metadata_dict["meta_val"])[:,1]

    return val_preds

def train_model(data, run_config, model_hyperparameters = default_model_hyperparameters, training_hyperparameters = default_training_hyperparameters):
        cnn_model = BASIC_CNN1D(num_classes=1, input_channels=8, **model_hyperparameters)
        cnn_model.to(device)

        criterion = torch.nn.BCEWithLogitsLoss()  # Includes sigmoid activation
        optimizer = torch.optim.SGD(cnn_model.parameters(), lr=training_hyperparameters["lr"], weight_decay=WEIGHT_DECAY)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=15)

        train_dataloader, val_dataloader, test_dataloader, metadata_dict = get_train_val_split_dataloaders(**data, run_config=run_config)

        training_losses = []
        val_losses, val_accuracies, pred_probas, pred_gts = [], [], [], []
        test_losses, test_accuracies, test_pred_probas, test_pred_gts = [], [], [], []

        for epoch in range(run_config["num_epochs"]):
            curr_epoch_avg_train_loss = train_loop(cnn_model, train_dataloader, loss_fn=criterion, optimizer=optimizer)
            curr_epoch_val_loss, curr_epoch_val_accuracy, curr_epoch_pred_probas, curr_epoch_pred_gts = test_loop(cnn_model, val_dataloader, loss_fn=criterion, scheduler=lr_scheduler)
            curr_epoch_test_loss, curr_epoch_test_accuracy, curr_epoch_test_pred_probas, curr_epoch_test_pred_gts = test_loop(cnn_model, test_dataloader, loss_fn=criterion, scheduler=None) # no scheduler for the test set

            logging.info(
                f"Epoch [{epoch + 1}/{run_config['num_epochs']}], average Validation Loss: {curr_epoch_val_loss:.4f}, Accuracy: {curr_epoch_val_accuracy:.4f}, lr: {optimizer.param_groups[0]['lr']}"
            )

            val_losses.append(curr_epoch_val_loss)
            training_losses.append(curr_epoch_avg_train_loss)
            val_accuracies.append(curr_epoch_val_accuracy)
            pred_probas.append(curr_epoch_pred_probas)
            pred_gts.append(curr_epoch_pred_gts)

            test_losses.append(curr_epoch_test_loss)
            test_accuracies.append(curr_epoch_test_accuracy)
            test_pred_probas.append(curr_epoch_test_pred_probas)
            test_pred_gts.append(curr_epoch_test_pred_gts)

        metadata_log_reg_model = LogisticRegression(C=0.01, max_iter=4000)
        # TODO what about the test set? Why does is this called val??
        metadata_val_preds = handle_metadata_log_model(metadata_log_reg_model, metadata_dict)

        performance_metrics = {
            "train_losses": training_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "val_pred_probas": pred_probas,
            "val_pred_gts": pred_gts,
            "metadata_val_pred": metadata_val_preds,
            "test_losses": test_losses,
            "test_accuracies": test_accuracies,
            "test_pred_probas": test_pred_probas,
            "test_pred_gts": test_pred_gts,
        }

        return performance_metrics

# %%
def run():
    print(f"Debug run started with properties: epochs: {run_config['num_epochs']}, repeats: {run_config['repeats']}")

    data = get_data_dict()
    performance_metrics_over_repeats = []
    for repeat in range(run_config['repeats']):
        print(f"repeat: {repeat}/{run_config['repeats']}", end="\r")

        # train_losses, val_losses, val_accuracies, roc/auc
        performance_metrics = train_model(data, run_config)
        performance_metrics_over_repeats.append(performance_metrics)

    evaluated_performance_metrics = evaluate_performance_metrics(performance_metrics_over_repeats)
    min_val_loss_dict, final_val_loss_dict, final_test_loss_dict = extract_final_accuracy_metrics(evaluated_performance_metrics)
    print(min_val_loss_dict)
    print(final_val_loss_dict)

    # Visualizations
    generate_plots(evaluated_performance_metrics, min_val_loss_dict["epoch_with_min_val_loss"], run_config=run_config)

    # sanity check
    # test_values()

# %%
if __name__ == "__main__":
    run()
# %%
