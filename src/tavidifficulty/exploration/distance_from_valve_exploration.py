# %%
%load_ext autoreload
%autoreload 2
from tavidifficulty.data.data_loading import get_extended_data, pad_if_necessary
from tavidifficulty.preprocessing.preprocessing_extended_data import remove_unfinished_sections, min_max_normalization_distance_from_valve
from tavidifficulty.data.parse_results_files import parse_all_results_and_targets_files_to_patients
from tavidifficulty.config import max_length_without_abdominal_aorta
import copy
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
# %%
patients = parse_all_results_and_targets_files_to_patients(remove_incomplete=True, remove_without_targets=True)
extended_dataframes = [patient.get_extended_params_without_abdominal_aorta for patient in patients]
# %%
max_len = 0

for df in extended_dataframes:
    if len(df) > max_len:
        max_len = len(df)

print(max_len)
# %%
current_columns_analysis = 'Distance from Valve[mm]'

'''
for patient in patients:
    df = patient.get_extended_params_without_abdominal_aorta[current_columns_analysis]
    plt.hist(df, bins=30)
    plt.title(f"hist of column: {current_columns_analysis}, patient: {patient.id}")
    plt.show()
'''
# %%
# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Auto-detect terminal width
pd.set_option('display.max_colwidth', None)  # Show full content of cells

def inspec_patient(pid):
    for patient in patients:
        if patient.id == pid:
            return patient.get_extended_params_without_abdominal_aorta
        
weird_ids = ["2082551"]

for weird_id in weird_ids:
    df = inspec_patient(weird_id)
    display(df)
# %%
'''Explanation: some slices have not been done yet. The distance to the valve is therefore zero. 
We fixed this by deleting the rows in the preprocessing function: remove_unfinished_sections'''


# %%
# now let's proceed to normalizing and centering the data.
# let's not overthink it, let's just get the max and min over the entire dataset.
extended_dataframes_without_unfinished_slices = [remove_unfinished_sections(extended_dataframe) for extended_dataframe in extended_dataframes]

global_min = 0
global_max = 0

for extended_dataframes_without_unfinished_slice in extended_dataframes_without_unfinished_slices:
    curr_min = extended_dataframes_without_unfinished_slice[current_columns_analysis].iloc[0]
    curr_max = extended_dataframes_without_unfinished_slice[current_columns_analysis].iloc[-1]
    
    if curr_min < global_min:
        global_min = curr_min

    print(curr_max)
    if curr_max > global_max:
        global_max = curr_max

print(global_min)
print(global_max)
# %%
extended_dataframes_without_unfinished_slices_copy = copy.deepcopy(extended_dataframes_without_unfinished_slices)
normalized_dfs = [min_max_normalization_distance_from_valve(extended_dataframes_without_unfinished_slice) for extended_dataframes_without_unfinished_slice in extended_dataframes_without_unfinished_slices_copy]

for (normalized_df, non_normalized_df) in zip(normalized_dfs, extended_dataframes_without_unfinished_slices):
    plt.hist(normalized_df[current_columns_analysis], bins = 100)
    plt.title("normalized ones")
    plt.show()

    plt.hist(non_normalized_df[current_columns_analysis], bins = 100)
    plt.title("non normalized ones")
    plt.show()
# %%
display(normalized_dfs[0])
display(extended_dataframes_without_unfinished_slices[0])
# %%
