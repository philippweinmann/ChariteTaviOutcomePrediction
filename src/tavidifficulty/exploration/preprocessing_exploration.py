# %%
from tavidifficulty.data.parse_results_files import parse_all_results_and_targets_files_to_patients
from tavidifficulty.data.patient import get_dataset
import numpy as np
from IPython.display import display
import pandas as pd
# %%
patients = np.array(parse_all_results_and_targets_files_to_patients(remove_incomplete=True, remove_outliers=False, remove_without_targets=True))
# %%
# we have some outliers, and the model seems to be able to handle them okay. However I would like to improve it by fixing them.
# One idea would be to do some automatic clipping and then automatic scaling.

# automatic, because there are simply too many features to do it manually.
X_no_scale_no_clip, y_no_scale_no_clip = get_dataset(patients, clip_values=False, scale=False)
X_no_scale, y_no_scale = get_dataset(patients, clip_values=True, percentile=0.05, scale=False)
X_scale, y_scale = get_dataset(patients, clip_values=True, percentile=0.05, scale=True)

data_rows = [X_no_scale_no_clip.iloc[0], X_no_scale.iloc[0], X_scale.iloc[0]]
indexes = ["No Scale No Clip", "No Scale", "Scale"]
# %%
# okay we need a way to visualize how the data has changed.
def visualize_data_change(data_rows, indexes):
    # create a dataframe with the first row of each dataframe
    df = pd.DataFrame(data_rows, index=indexes)
    display(df)
    return df

df = visualize_data_change(data_rows, indexes)
# %%
