# %%
from tavidifficulty.data.parse_results_files import parse_all_results_and_targets_files_to_patients
from tavidifficulty.data.patient import get_dataset
import numpy as np
import matplotlib.pyplot as plt
# %%
patients = np.array(parse_all_results_and_targets_files_to_patients(remove_incomplete=True, remove_outliers=False, remove_without_targets=True))
# %%
pos_outcome = len([p for p in patients if p.optimal])
neg_outcome = len([p for p in patients if not p.optimal])

plt.bar(['Positive', 'Negative'], [pos_outcome, neg_outcome])
plt.title("patients with positive or negative TAVI procedure outcomes")
plt.show()
# %%
device_se = len([p for p in patients if p.se_device])
device_be = len([p for p in patients if not p.se_device])

plt.bar(['Self expanding', 'Balloon expanding'], [device_se, device_be])
plt.title("Balloons or self expanding devices used in TAVI procedures")
plt.show()
# %%
