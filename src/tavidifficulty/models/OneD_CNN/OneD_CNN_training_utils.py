# %%
import pprint
import pickle
from datetime import datetime

from sklearn.metrics import f1_score, classification_report
from tavidifficulty.config import training_logs_folder
from tavidifficulty.models.OneD_CNN.OneD_CNN_config import label, modification, default_model_hyperparameters, default_training_hyperparameters
from tavidifficulty.visualization.visualization_utils import visualize_f1_scores, generate_roc_curve_for_best_epoch, visualize_training_logs, visualize_accuracies, visualize_progress
import logging
import os
import numpy as np
# %%
def save_training_logs(training_losses, validation_losses, label:str=label, modification:str=modification, use_label_as_filename = False):
    current_datetime = datetime.now()
    datetime_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    # Format it as a string suitable for a filename
    if not use_label_as_filename:
        filename = datetime_str + ".pkl"
    else:
        filename = label.replace(" ", "_").strip() + ".pkl"

    logs_dict = {
        "timestamp": current_datetime,
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "label": label,
        "modification": modification,
    }

    training_logs_fp = training_logs_folder / filename

    with open(training_logs_fp, mode="wb") as f:
        pickle.dump(logs_dict, f)

    logging.info(f"dumped the training logs as a dict to: {training_logs_fp}")

    return datetime_str


def aggregate_raw_model_outputs(model_outputs_over_repeat):
    # note that this method can also handle any sort of array data, that has the same shape as expected from model outputs over repeat.
    # in particulat ground truths

    # expected input shape: num_repeats x num_epochs x num_validation_values

    # 1. let's combine each repeat to get one list of list of probas per epoch
    model_outputs_aggregated = zip(*model_outputs_over_repeat)
    # 2. let's combine each list of probas to one big list using ravel
    model_outputs_aggregated = [np.array(pred_probas_aggregated_curr_epoch).ravel() for pred_probas_aggregated_curr_epoch in model_outputs_aggregated]

    # shape: num_epochs x (num_val_values * repeats)
    return model_outputs_aggregated


def evaluate_performance_metrics(performance_metrics_over_repeats):
    train_losses_over_repeat = []
    val_losses_over_repeat, val_accuracies_over_repeat, val_pred_probas_over_repeat, val_pred_gts_over_repeat = [], [], [], []
    test_losses_over_repeat, test_accuracies_over_repeat, test_pred_probas_over_repeat, test_pred_gts_over_repeat = [], [], [], []

    for performance_metric_over_repeat in performance_metrics_over_repeats:
        train_losses_over_repeat.append(performance_metric_over_repeat["train_losses"])

        val_losses_over_repeat.append(performance_metric_over_repeat["val_losses"])
        val_accuracies_over_repeat.append(performance_metric_over_repeat["val_accuracies"])
        val_pred_probas_over_repeat.append(performance_metric_over_repeat["val_pred_probas"])
        val_pred_gts_over_repeat.append(performance_metric_over_repeat["val_pred_gts"])

        test_losses_over_repeat.append(performance_metric_over_repeat["test_losses"])
        test_accuracies_over_repeat.append(performance_metric_over_repeat["test_accuracies"])
        test_pred_probas_over_repeat.append(performance_metric_over_repeat["test_pred_probas"])
        test_pred_gts_over_repeat.append(performance_metric_over_repeat["test_pred_gts"])

    # let's average the losses.
    train_losses_ageraged = [sum(col) / len(col) for col in zip(*train_losses_over_repeat)]
    val_losses_ageraged = [sum(col) / len(col) for col in zip(*val_losses_over_repeat)]
    val_accuracies_aggregated = [sum(col) / len(col) for col in zip(*val_accuracies_over_repeat)]

    test_losses_ageraged = [sum(col) / len(col) for col in zip(*test_losses_over_repeat)]
    test_accuracies_aggregated = [sum(col) / len(col) for col in zip(*test_accuracies_over_repeat)]

    # roc/auc
    # since we calculate only one roc per epoch, we transform the values to the following shape: num_epochs x (num_val_values * repeats)
    val_pred_probas_aggregated = aggregate_raw_model_outputs(val_pred_probas_over_repeat)
    val_pred_gts_aggregated = aggregate_raw_model_outputs(val_pred_gts_over_repeat)

    test_pred_probas_aggregated = aggregate_raw_model_outputs(test_pred_probas_over_repeat)
    test_pred_gts_aggregated = aggregate_raw_model_outputs(test_pred_gts_over_repeat)

    # f1 scores:
    val_f1_scores = [f1_score(np.array(val_pred_proba_aggregated) > 0.5, val_pred_gt_aggregated) for val_pred_proba_aggregated, val_pred_gt_aggregated in zip(val_pred_probas_aggregated, val_pred_gts_aggregated)]
    test_f1_scores = [f1_score(np.array(test_pred_proba_aggregated) > 0.5, test_pred_gt_aggregated) for test_pred_proba_aggregated, test_pred_gt_aggregated in zip(test_pred_probas_aggregated, test_pred_gts_aggregated)]

    # variance scores:
    val_acc_variances = [np.array(col).std() for col in zip(*val_losses_over_repeat)]
    test_acc_variances = [np.array(col).std() for col in zip(*test_losses_over_repeat)]

    evaluated_performance_metrics = {
        "train_losses_ageraged": np.array(train_losses_ageraged),
        "val_f1_scores": np.array(val_f1_scores),
        "test_f1_scores": np.array(test_f1_scores),
        "val_losses_ageraged": np.array(val_losses_ageraged),
        "val_accuracies_aggregated": np.array(val_accuracies_aggregated),
        "val_pred_probas_aggregated": np.array(val_pred_probas_aggregated),
        "val_pred_gts_aggregated": np.array(val_pred_gts_aggregated),
        "val_acc_variances": np.array(val_acc_variances),
        "test_losses_ageraged": np.array(test_losses_ageraged),
        "test_accuracies_aggregated": np.array(test_accuracies_aggregated),
        "test_pred_probas_aggregated": np.array(test_pred_probas_aggregated),
        "test_pred_gts_aggregated": np.array(test_pred_gts_aggregated),
        "test_acc_variances": np.array(test_acc_variances),
    }

    return evaluated_performance_metrics

# %%
hyperparameter_filename = "hyperparameters.txt"

def save_hyperparameters_to_text(logging_folder):
    comb_hyperparameter_dict = default_model_hyperparameters.copy()
    comb_hyperparameter_dict.update(default_training_hyperparameters)

    fp = logging_folder / hyperparameter_filename
    with open(fp, mode="w") as f:
        f.write(str(comb_hyperparameter_dict))
        logging.info(f"saved hyperparameter dict of current run at: {fp}")

def save_dicts_to_text(logging_folder, filename, dictionaries):
    fp = logging_folder / filename

    formatter = pprint.PrettyPrinter(indent=4, width=80, depth=None, sort_dicts=False)

    with open(fp, mode="w") as f:
        for idx, dictionary in enumerate(dictionaries):
            # Add separation between dictionaries if multiple exist
            if idx > 0:
                f.write("\n\n")
            f.write(formatter.pformat(dictionary))
        
        logging.info(f"saved dictionaries of current run at: {fp}")
    
    return fp

def add_classification_report_to_metrics(file, evaluated_performance_metrics_dict, epoch_at_min_val_loss):
    val_gts = evaluated_performance_metrics_dict['val_pred_gts_aggregated'][epoch_at_min_val_loss]
    val_bin_preds = evaluated_performance_metrics_dict['val_pred_probas_aggregated'][epoch_at_min_val_loss] > 0.5
    val_class_report = classification_report(val_gts, val_bin_preds, target_names=["not optimal", "optimal"])
    
    test_gts = evaluated_performance_metrics_dict['test_pred_gts_aggregated'][epoch_at_min_val_loss]
    test_bin_preds = evaluated_performance_metrics_dict['test_pred_probas_aggregated'][epoch_at_min_val_loss] > 0.5
    test_class_report = classification_report(test_gts, test_bin_preds, target_names=["not optimal", "optimal"])

    with open(file, mode="a") as f:
        f.write(f"\n\n\nval classification report at min val loss: {val_class_report}")
        f.write(f"\ntest classification report at min val loss: {test_class_report}")



# %%
def get_metrics_dict_at_epoch(evaluated_performance_metrics_dict, epoch, descriptor):
    min_val_loss_dict = {
        "descriptor": descriptor,
        "epoch": epoch,
        "val_loss": evaluated_performance_metrics_dict['val_losses_ageraged'][epoch],
        "val_acc": evaluated_performance_metrics_dict['val_accuracies_aggregated'][epoch],
        "val_acc_var": evaluated_performance_metrics_dict['val_acc_variances'][epoch],
        "test_loss": evaluated_performance_metrics_dict['test_losses_ageraged'][epoch],
        "test_acc": evaluated_performance_metrics_dict['test_accuracies_aggregated'][epoch],
        "test_acc_var": evaluated_performance_metrics_dict['test_acc_variances'][epoch],
    }

    return min_val_loss_dict

def extract_final_accuracy_metrics(evaluated_performance_metrics):
    min_val_loss_idx = evaluated_performance_metrics['val_losses_ageraged'].argmin()
    metrics_at_min_val_loss_descriptor = "performance metrics at the epoch, where the validation loss was minimal"
    metrics_at_min_val_loss = get_metrics_dict_at_epoch(evaluated_performance_metrics, epoch=min_val_loss_idx, descriptor=metrics_at_min_val_loss_descriptor)

    metrics_at_last_epoch_descriptor = "performance metrics at epoch 99, currently the last epoch"
    metrics_at_last_epoch_dict = get_metrics_dict_at_epoch(evaluated_performance_metrics, epoch=99, descriptor=metrics_at_last_epoch_descriptor)

    return metrics_at_min_val_loss, metrics_at_last_epoch_dict

def generate_plots(evaluated_performance_metrics, epoch_with_min_val_loss, run_config, log_folder_fp=None):
    # Visualizations
    train_logs_fig, _ = visualize_training_logs(training_losses=evaluated_performance_metrics['train_losses_ageraged'], validation_losses=evaluated_performance_metrics['val_losses_ageraged'], test_losses=evaluated_performance_metrics['test_losses_ageraged'], title = f"aggregated losses over {run_config['repeats']} repetitions")
    accuracy_fig, _ = visualize_accuracies(accuracies=[evaluated_performance_metrics['val_accuracies_aggregated'], evaluated_performance_metrics['test_accuracies_aggregated']], labels = ["validation set accuracies", "test set accuracies"], title="validation accuracy visualization over training")
    
    val_roc_curve_fig = generate_roc_curve_for_best_epoch(evaluated_performance_metrics['val_pred_probas_aggregated'], evaluated_performance_metrics['val_pred_gts_aggregated'], epoch=epoch_with_min_val_loss)
    test_roc_curve_fig = generate_roc_curve_for_best_epoch(evaluated_performance_metrics['test_pred_probas_aggregated'], evaluated_performance_metrics['test_pred_gts_aggregated'], epoch=epoch_with_min_val_loss)

    f1_scores_fig, _ = visualize_f1_scores(val_f1_scores = evaluated_performance_metrics['val_f1_scores'], test_f1_scores = evaluated_performance_metrics['test_f1_scores'])

    progress_fig, _ = visualize_progress()
    
    if log_folder_fp:
        # save the plots
        train_logs_fig.savefig(str(log_folder_fp / "train_logs_fig"))
        accuracy_fig.savefig(str(log_folder_fp / "accuracies_logs_fig"))
        progress_fig.savefig(str(log_folder_fp / "progress_fig"))
        val_roc_curve_fig.savefig(str(log_folder_fp / ("val_roc_curve_fig")))
        test_roc_curve_fig.savefig(str(log_folder_fp / ("test_roc_curve_fig")))
        f1_scores_fig.savefig(str(log_folder_fp / "f1_scores_fig"))
    else:
        logging.warning("log folder not specified, not saving the plots, but showing them directly")
        progress_fig.show()
        train_logs_fig.show()
        accuracy_fig.show()
        val_roc_curve_fig.show()
        test_roc_curve_fig.show()
        f1_scores_fig.show()

def end_of_run_logging(performance_metrics_over_repeats, run_config):
    evaluated_performance_metrics = evaluate_performance_metrics(performance_metrics_over_repeats)
    
    metrics_at_min_val_loss, metrics_at_last_epoch_dict = extract_final_accuracy_metrics(evaluated_performance_metrics)
    epoch_at_min_val_loss = metrics_at_min_val_loss["epoch"]

    # save the training logs
    training_logs_datetime_str = save_training_logs(training_losses=evaluated_performance_metrics['train_losses_ageraged'], validation_losses=evaluated_performance_metrics['val_losses_ageraged'])
    
    # this is a separate folder, where all the logs and plots are saved.
    log_folder_fp = training_logs_folder / training_logs_datetime_str
    os.makedirs(log_folder_fp)

    generate_plots(evaluated_performance_metrics, epoch_with_min_val_loss=epoch_at_min_val_loss, log_folder_fp=log_folder_fp, run_config=run_config)

    # save the hyperparameters
    save_hyperparameters_to_text(log_folder_fp)

    # save results
    metrics_fp = save_dicts_to_text(log_folder_fp, filename="metrics.txt", dictionaries=[metrics_at_min_val_loss, metrics_at_last_epoch_dict])

    # save classification reports to text as well
    add_classification_report_to_metrics(metrics_fp, evaluated_performance_metrics, epoch_at_min_val_loss=epoch_at_min_val_loss)

def prepare_data_for_model_input(dataframe):
    dataframe = dataframe.to_numpy()
    dataframe = np.transpose(dataframe)

    return dataframe
