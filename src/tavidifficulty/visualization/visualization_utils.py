import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import pickle
from sklearn.metrics import roc_auc_score, roc_curve

from tavidifficulty.config import training_logs_folder


def get_all_training_logs():
    training_logs = []

    for entry in os.scandir(training_logs_folder):
        if entry.is_file():
            file_path = training_logs_folder / entry.name

            with open(file_path, mode="rb") as f:
                training_log = pickle.load(f)

            training_logs.append(training_log)

    # sort by timestamp
    training_logs = sorted(training_logs, key=lambda log: log['timestamp'])

    return training_logs

def visualize_progress():
    # Create figure and axes objects explicitly
    fig, ax = plt.subplots(figsize=(10, 6))
    
    training_logs = get_all_training_logs()
    
    # Plot the most recent training log
    most_recent_training_log = training_logs[-1]
    epochs_last_training_log = np.arange(1, len(most_recent_training_log["training_losses"]) + 1)
    
    lines = []
    # Plot using axes object instead of plt
    latest_train_line, = ax.plot(
        epochs_last_training_log, 
        np.clip(most_recent_training_log["training_losses"], a_min=None, a_max=0.8),
        color="black", 
        linewidth=2,
        label=most_recent_training_log["label"] + " (training)",
        linestyle="--"
    )
    latest_val_line, = ax.plot(
        epochs_last_training_log, 
        most_recent_training_log["validation_losses"],
        color="orange", 
        linewidth=2,
        label=most_recent_training_log["label"] + " (validation)"
    )
    lines.append(latest_train_line)
    lines.append(latest_val_line)

    # Plot previous training logs
    training_logs = training_logs[:-1]
    grays = np.linspace(0.2, 0.8, len(training_logs))
    
    for idx, (training_log, gray) in enumerate(zip(training_logs, grays)):
        val_line, = ax.plot(
            np.arange(1, len(training_log["training_losses"]) + 1),
            training_log["validation_losses"],
            color= str(gray), 
            alpha= 0.7,
            label= str(idx) + ". " + training_log["label"] + " (validation)"
        )
        lines.append(val_line)

    # Configure plot properties
    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Development Progress Visualized via Loss Progression", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Create legend and set up interactivity
    leg = ax.legend(loc='upper right', fontsize=9)
    pickradius = 5  # How close click needs to be to trigger event

    # Map legend lines to plot lines
    map_legend_to_ax = {}
    for legend_line, ax_line in zip(leg.get_lines(), lines):
        legend_line.set_picker(pickradius)
        map_legend_to_ax[legend_line] = ax_line

    # Event handler for legend clicks
    def on_pick(event):
        legend_line = event.artist
        if legend_line not in map_legend_to_ax:
            return
            
        ax_line = map_legend_to_ax[legend_line]
        visible = not ax_line.get_visible()
        ax_line.set_visible(visible)
        legend_line.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw_idle()  # Update canvas using figure reference

    # Connect event handler to figure
    fig.canvas.mpl_connect('pick_event', on_pick)
    
    return fig, ax


def visualize_training_logs(training_losses, validation_losses, test_losses, title=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    epochs = np.arange(1, len(training_losses) + 1)
    ax.plot(epochs, validation_losses, label="Validation Loss", color="orange", linewidth=2)
    ax.plot(epochs, test_losses, label="Test Loss", color="red", linewidth=2)
    ax.plot(epochs, np.clip(training_losses, a_min=None, a_max=0.8), 
            label="Training Loss", color="gray", linestyle="--")
    
    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title or "Training Logs Visualization", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.6)
    
    return fig, ax

# Minimalist version keeping original style
def visualize_accuracies(accuracies, labels, title):
    fig, ax = plt.subplots()
    epochs = np.arange(1, len(accuracies[0]) + 1)
    colours = ["orange", "red"] # I know bad code
    for idx, (accuracy, label) in enumerate(zip(accuracies, labels)):
        ax.plot(epochs, accuracy, 
                label=label, 
                color=colours[idx])
    ax.set_xlabel("epochs")
    ax.set_ylabel("accuracy (percentage)")
    ax.set_title(title)
    ax.grid(True, alpha=0.6)
    ax.legend()
    return fig, ax

def visualize_grid_training_logs(train_losses_over_repeat, val_losses_over_repeat):
    plt.figure()

    amt_repeats = len(train_losses_over_repeat)
    if amt_repeats > 9:
        logging.warning("too many logs to display them all. Selecting the first 9")

        val_losses_over_repeat = val_losses_over_repeat[0:9]
        train_losses_over_repeat = train_losses_over_repeat[0:9]

    n_cols = min(3, amt_repeats)
    n_rows = int(np.ceil(amt_repeats / 3))
    
    fig = plt.figure()

    for i, (train_losses, val_losses) in enumerate(zip(train_losses_over_repeat, val_losses_over_repeat)):
        epochs = np.arange(1, len(train_losses) + 1)
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        ax.plot(epochs, val_losses, label="val losses", color="orange")
        ax.plot(epochs, train_losses, label="train losses", color="gray")
        
        # ax.set_xlabel("epochs")
        # ax.set_ylabel("loss")
        # ax.legend()

        for spine in ax.spines.values():
            spine.set_visible(True)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()

def visualize_roc(ns_fpr, ns_tpr, lr_fpr, lr_tpr, lr_auc, epoch):
    fig, ax = plt.subplots(figsize=(10,6))

    # plot the roc curve for the model
    ax.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    ax.plot(lr_fpr, lr_tpr, label='One dimensional CNN, ROC curve (area = %0.2f)' % lr_auc)
    # axis labels
    ax.set_xlabel('False Positive Rate', fontsize=16)
    ax.set_ylabel('True Positive Rate', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    # show the legend
    ax.legend(fontsize=16)

    ax.set_title(f"ROC curve for the one dimensional CNN done at epoch: {epoch}", fontsize=20)
    
    return fig, ax

def generate_roc_curve(curr_epoch_pred_probas, curr_epoch_pred_gts, epoch):
    ns_probs = [0 for _ in range(len(curr_epoch_pred_gts))]

    # calculate scores
    ns_auc = roc_auc_score(curr_epoch_pred_gts, ns_probs)
    lr_auc = roc_auc_score(curr_epoch_pred_gts, curr_epoch_pred_probas)

    # summarize scores
    logging.info('No Skill: ROC AUC=%.3f' % (ns_auc))
    logging.info('Logistic: ROC AUC=%.3f' % (lr_auc))

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(curr_epoch_pred_gts, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(curr_epoch_pred_gts, curr_epoch_pred_probas)

    fig, _ = visualize_roc(ns_fpr, ns_tpr, lr_fpr, lr_tpr, lr_auc, epoch)

    return fig

def generate_roc_curve_for_best_epoch(pred_probas_aggregated, pred_gts_aggregated, epoch):
    curr_epoch_probas = pred_probas_aggregated[epoch]
    curr_epoch_gts = pred_gts_aggregated[epoch]

    fig = generate_roc_curve(curr_epoch_pred_probas=curr_epoch_probas, curr_epoch_pred_gts=curr_epoch_gts, epoch=epoch)

    return fig

def visualize_f1_scores(val_f1_scores, test_f1_scores):
    fig, ax = plt.subplots(figsize=(10,6))
    epochs = np.arange(1, len(val_f1_scores) + 1)
    ax.plot(epochs, val_f1_scores, 
            label="val f1 scores (aggregated)", 
            color="orange")
    ax.plot(epochs, test_f1_scores, 
            label="test f1 scores (aggregated)", 
            color="red")
    ax.set_xlabel("epochs")
    ax.set_ylabel("f1 scores")
    ax.set_title("f1 score visualization over training")
    ax.grid(True, alpha=0.6)
    ax.legend()
    return fig, ax