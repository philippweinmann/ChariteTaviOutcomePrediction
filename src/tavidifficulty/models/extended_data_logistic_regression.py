# %%
import numpy as np
from tavidifficulty.data.parse_results_files import parse_all_results_and_targets_files_to_patients
from tavidifficulty.data.patient import get_dataset

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
# %%
# 1. get dataset
patients = np.array(parse_all_results_and_targets_files_to_patients(use_old_targets = True, use_new_targets = False, remove_incomplete=True, remove_outliers=False, remove_without_targets=True))
X, y = get_dataset(patients)
# %%
# 2. define model
def get_model(C=0.01, max_iter=4000):
    model = LogisticRegression(C=C, max_iter=max_iter)
    return model

example_model = get_model()
# %%
def get_predictions():
    model = get_model()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:,1]
    gts = y_test

    return preds, gts

# TODO 1000 is the correct number for reliable results
n_repeats = 1000
agg_preds, agg_gts = [], []

for r in range(n_repeats):
    preds, gts = get_predictions()
    agg_preds.append(preds)
    agg_gts.append(gts)

agg_preds = np.array(agg_preds).ravel()
bin_preds = agg_preds > 0.5
agg_gts = np.array(agg_gts).ravel()

print("Classification report_\n", classification_report(agg_gts, bin_preds))
# %%
# Now the ROC/AUC curves
ns_probs = [0 for _ in range(len(agg_gts))]

# calculate scores
ns_auc = roc_auc_score(agg_gts, ns_probs)
lr_auc = roc_auc_score(agg_gts, agg_preds)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(agg_gts, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(agg_gts, agg_preds)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, label='Logistic Regression, ROC curve (area = %0.2f)' % lr_auc)
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC curve for log regression model")
plt.legend()
# show the plot
plt.show()
# %%
