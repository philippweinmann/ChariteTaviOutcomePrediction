# %%
%load_ext autoreload
%autoreload 2
from tavidifficulty.data.data_loading import get_extended_data, pad_if_necessary
from tavidifficulty.preprocessing.preprocessing_extended_data import remove_unfinished_sections, interpolate_zero_series
from tavidifficulty.data.parse_results_files import parse_all_results_and_targets_files_to_patients
from tavidifficulty.config import max_length_without_abdominal_aorta
import copy
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from tavidifficulty.preprocessing.preprocessing_utils import Winsorizer
from tavidifficulty.exploration.exploration_utils import visualize_combined_column
# %%
patients = parse_all_results_and_targets_files_to_patients(remove_incomplete=True, remove_without_targets=True)
extended_dataframes = [patient.get_extended_params_without_abdominal_aorta for patient in patients]

current_column = "Crosssection Area[mm2]"
visualize_combined_column(extended_dataframes, column_name=current_column)
# %%
# fixing steps to apply:
# 1. remove unfinished sections

# transformation steps to apply:
# 1. MaxAbsScale Distance from valve

# we copy these functions to get a snapshot of how they looked at the time of exploration
from tavidifficulty.preprocessing.preprocessing_config import MIN_DISTANCE_TO_VALVE, MAX_DISTANCE_TO_VALVE, MIN_CROSSSECTION, MAX_CROSSSECTION
import pandas as pd

from sklearn.preprocessing import MaxAbsScaler

# distance from valve
def remove_unfinished_sections(dataframe):
    reverse_df = dataframe[::-1]

    for i in range(len(reverse_df)):
        if reverse_df["Distance from Valve[mm]"].iloc[i] != 0:
            cutoff_i = reverse_df.index[i]
            break
    
    cuttoff_dataframe = dataframe[dataframe.index <= cutoff_i]

    return cuttoff_dataframe

def train_apply_scaler(training_dataframes, non_training_dataframes, scaler, columnName):
    combined_train_column = pd.concat([df[[columnName]] for df in training_dataframes])
    scaler.fit(combined_train_column)

    for dataframe in training_dataframes:
        transformed_col = scaler.transform(dataframe[[columnName]]).ravel()
        dataframe.loc[:,columnName] = transformed_col

    for dataframe in non_training_dataframes:
        transformed_col = scaler.transform(dataframe[[columnName]]).ravel()
        dataframe.loc[:,columnName] = transformed_col

    return training_dataframes, non_training_dataframes


def train_scaler(dataframes, columnName, scaler):
    combined_column = pd.concat([df[[columnName]] for df in dataframes])
    scaler.fit(combined_column)

    return scaler

def apply_previous_preprocessing_functions(dataframes):
    dataframes = [remove_unfinished_sections(dataframe) for dataframe in dataframes]

    dataframes, _ = train_apply_scaler(dataframes, [], scaler=MaxAbsScaler(), columnName="Distance from Valve[mm]")

    return dataframes
# %%

df_pp = apply_previous_preprocessing_functions(extended_dataframes)
original_df_pp = copy.deepcopy(df_pp)
# %%
print(df_pp[0].columns)
# %%
display(df_pp[0][current_column])
# %%
def plot_hist(patient):
    id = patient.id
    dataframe = apply_previous_preprocessing_functions([patient.get_extended_params_without_abdominal_aorta])[0]

    col = dataframe[current_column]
    plt.hist(col, bins=400)
    plt.xlabel("bins")
    plt.ylabel("amt of elements in bin")
    plt.title(f"histogram of {current_column} for patient with id: {id}")
    plt.show()

'''
for patient in patients:
    plot_hist(patient)
'''
# %%
interesting_ids = ["2079752", "2075420"]

# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Auto-detect terminal width
pd.set_option('display.max_colwidth', None)  # Show full content of cells

for patient in patients:
    if str(patient.id) in interesting_ids:
        print(str(patient.id))
        display(apply_previous_preprocessing_functions([patient.get_extended_params_without_abdominal_aorta])[0])
# %%
suspicious_ids = []
for patient in patients:
    dataframe = apply_previous_preprocessing_functions([patient.get_extended_params_without_abdominal_aorta])[0]
    amt_zeroes = (dataframe[current_column] == 0).sum()

    if amt_zeroes > 0:
        print(patient.id)
        suspicious_ids.append(patient.id)
# %%
sus_patients = [patient for patient in patients if patient.id in suspicious_ids]
for sus_patient in sus_patients:
    plot_hist(sus_patient)
# %%
# okay let's think about this. The crosssection is the area. It does not just go to zero.
# I will try to just assume that the value will go from the previous non zero value to the next non zero value.
def interpolate_zero_series(df):
    df_copy = copy.copy(df)
    # Check if the first or last value is zero
    if df_copy[current_column].iloc[0] == 0:
        raise ValueError("The first value in column 'X' is zero.")
    if df_copy[current_column].iloc[-1] == 0:
        raise ValueError("The last value in column 'X' is zero.")
    
    # Create a mask to identify zero values
    zero_mask = df_copy[current_column] == 0
    # Replace zeros with NaN to prepare for interpolation
    df_copy.loc[zero_mask, current_column] = pd.NA
    # Perform linear interpolation
    df_copy[current_column] = df_copy[current_column].interpolate(method='linear')
    return df_copy

def plot_hist_from_df(dataframe):
    print(current_column)
    col = dataframe[current_column]
    plt.hist(col, bins=400)
    plt.xlabel("bins")
    plt.ylabel("amt of elements in bin")
    plt.title(f"histogram of {current_column} for patient with id: {id}")
    plt.show()

sus_patient_dataframe = apply_previous_preprocessing_functions([sus_patients[0].get_extended_params_without_abdominal_aorta])[0]
plot_hist_from_df(sus_patient_dataframe)

interpolated_df = interpolate_zero_series(sus_patient_dataframe)
plot_hist_from_df(interpolated_df)
# %%
# okay there seems to be an outlier issue. Let's try to fix this using Winsorizing
winsorized_df = copy.deepcopy(interpolated_df)
winsorized_df, _ = train_apply_scaler([winsorized_df], [], scaler=Winsorizer(lower=0.03, upper=0.97), columnName="Crosssection Area[mm2]")

plot_hist_from_df(winsorized_df[0])
# %%