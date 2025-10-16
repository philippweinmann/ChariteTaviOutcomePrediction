# %%
from tavidifficulty.data.parse_results_files import parse_all_results_and_targets_files_to_patients
import numpy as np

patients = parse_all_results_and_targets_files_to_patients(remove_incomplete=True, remove_without_targets=True)

# %%
# let's filter out the patients that do not have euroscores.
patients_wes = [patient for patient in patients if patient.preop_euroscore_ii]
print(len(patients_wes)) # they all do, cool

for patient in patients_wes:
    print(patient.preop_euroscore_ii)
# %%
euroscores = np.array([patient.preop_euroscore_ii for patient in patients_wes])
euroscores = euroscores / 100


print(euroscores.min())
print(euroscores.mean())
print(euroscores.max())

# euroscores above 7%:
print((euroscores > 7).sum())
# %%
# let's count the amount of patients that would've been saved, if we refused to make the operation given a too high euroscore (>7)
threshold = 7
patients_pot_saved = 0
false_positives = 0
for patient in patients_wes:
    if (not patient.optimal) & ((patient.preop_euroscore_ii / 100) > threshold):
        patients_pot_saved += 1

    if patient.optimal & ((patient.preop_euroscore_ii / 100) > threshold):
        false_positives += 1

print(patients_pot_saved)
print(false_positives)
# %%
# let's do this properly and let's create a classification report for euroscores 2
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

threshold = 7
euroscores = np.array([patient.preop_euroscore_ii for patient in patients_wes])
euroscores = euroscores / 100


operation_optimal = [patient.optimal for patient in patients_wes]

def print_class_report_for_euroscores(threshold):
    euroscore_below_threshold = euroscores < threshold
    print(f"threshold: {threshold}: \n{classification_report(operation_optimal, euroscore_below_threshold)}")
    print(f"confusion matrix: {confusion_matrix(operation_optimal, euroscore_below_threshold)}")

    accuracy = accuracy_score(operation_optimal, euroscore_below_threshold)
    return accuracy

# 7 would be the threshold used for not
print_class_report_for_euroscores(threshold=7)
# %%
# let's also find the best accuracy threshold
import optuna
from optuna.visualization import plot_slice

def opt_func(trial):
    threshold_sugg = trial.suggest_float("threshold", 2, 20)
    acc = print_class_report_for_euroscores(threshold=threshold_sugg)

    return acc

# study = optuna.create_study(direction="minimize")
# study.optimize(opt_func, n_jobs=-1, n_trials=500)

# print(f"best parameters: {study.best_params}")
# print(f"best value: {study.best_value}")

# %%
# plot_slice(study, target_name="accuracy of euroscore 2")
# %%
print_class_report_for_euroscores(threshold=7)
print_class_report_for_euroscores(threshold=8)
# %%
