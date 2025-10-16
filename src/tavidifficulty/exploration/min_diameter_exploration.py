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
from tavidifficulty.exploration.exploration_utils import visualize_combined_column, compare_visualize_combined_column
# %%
patients = parse_all_results_and_targets_files_to_patients(remove_incomplete=True, remove_without_targets=True)
extended_dataframes = [patient.get_extended_params_without_abdominal_aorta for patient in patients]

print(extended_dataframes[0].columns)
# %%
columnName = 'Min Diameter[mm]'
visualize_combined_column(extended_dataframes, columnName)
# %%
# let's apply previous pp steps
# let's apply the current preprocessing functions to see the impact
import pandas as pd
import logging
from tavidifficulty.preprocessing.preprocessing_utils import Winsorizer

from sklearn.preprocessing import MaxAbsScaler, StandardScaler, RobustScaler

# distance from valve
def remove_unfinished_sections(dataframe):
    reverse_df = dataframe[::-1]

    for i in range(len(reverse_df)):
        if reverse_df["Distance from Valve[mm]"].iloc[i] != 0:
            cutoff_i = reverse_df.index[i]
            break
    
    cuttoff_dataframe = dataframe[dataframe.index <= cutoff_i]

    return cuttoff_dataframe

# cross section
def interpolate_zero_series(dataframe, column_name):
    df_copy = dataframe.copy()
    # Check if the first or last value is zero
    if df_copy[column_name].iloc[0] == 0:
        raise ValueError(f"The first value in column {column_name} is zero.")
    if df_copy[column_name].iloc[-1] == 0:
        raise ValueError(f"The last value in column {column_name} is zero.")
    
    # Create a mask to identify zero values
    zero_mask = df_copy[column_name] == 0
    # Replace zeros with NaN to prepare for interpolation
    df_copy.loc[zero_mask, column_name] = pd.NA
    # Perform linear interpolation
    df_copy[column_name] = df_copy[column_name].interpolate(method='linear')

    return df_copy

def train_apply_transformation(training_dataframes, non_training_dataframes, scaler, columnName):
    combined_train_column = pd.concat([df[[columnName]] for df in training_dataframes])
    scaler.fit(combined_train_column)

    for dataframe in training_dataframes:
        transformed_col = scaler.transform(dataframe[[columnName]]).ravel()
        dataframe.loc[:,columnName] = transformed_col

    for dataframe in non_training_dataframes:
        transformed_col = scaler.transform(dataframe[[columnName]]).ravel()
        dataframe.loc[:,columnName] = transformed_col

    return training_dataframes, non_training_dataframes
# %%
# 1. fixing preprocessing step: remove unfinished sections
extended_dataframes_copy = copy.deepcopy(extended_dataframes)
extended_dataframes_copy = [remove_unfinished_sections(dataframe) for dataframe in extended_dataframes_copy]
# notice that we fixed issues in columns "distance from valve" and "distance from arch center", .
# %%
# 2. fixing preprocessing step: interpolate zero series
extended_dataframes_copy = [interpolate_zero_series(dataframe, column_name="Crosssection Area[mm2]") for dataframe in extended_dataframes_copy]
# %%
# 3. fixing preprocessing step: interpolate column area
extended_dataframes_copy = [interpolate_zero_series(dataframe, column_name="Area Diameter[mm]") for dataframe in extended_dataframes_copy]

# %%
visualize_combined_column(extended_dataframes_copy, columnName)
# %%
# okay there is still an issue with the zeroes.
# let's find the patients with zeroes

def show_all_dfs():
    for i, dataframe in enumerate(extended_dataframes_copy):
        col = dataframe[columnName]
        plt.hist(col, bins=40)
        plt.title(f"index: {i}")
        plt.show()

# show_all_dfs()
# %%
pd.set_option('display.max_rows', None)  # Show all rows
# display(extended_dataframes_copy[102])
# %%
indexes = [90,88,89,80,73,72]

# Define the column to highlight and the background color
highlight_color = '#290536'  # Pale yellow

for index in indexes:
    # Apply styling using the Styler object
    hightlighted_df = extended_dataframes_copy[index].style.apply(
        lambda col: [f'background-color: {highlight_color}' if col.name == columnName else '' 
                    for _ in col]
    )
    # display(hightlighted_df)
# %%
# okay, there are some extremely small values I need to get rid of. Anything below 0.5 I can get rid of I think.

# %%
def interpolate_below_value_series(dataframe, column_name, min_value):
    df_copy = dataframe.copy()
    # Check if the first or last value is zero
    if df_copy[column_name].iloc[0] < min_value:
        raise ValueError(f"The first value in column {column_name} is below {min_value}.")
    if df_copy[column_name].iloc[-1] < min_value:
        raise ValueError(f"The last value in column {column_name} is below {min_value}.")
    
    # Create a mask to identify zero values
    zero_mask = df_copy[column_name] < min_value
    # Replace zeros with NaN to prepare for interpolation
    df_copy.loc[zero_mask, column_name] = pd.NA
    # Perform linear interpolation
    df_copy[column_name] = df_copy[column_name].interpolate(method='linear')

    return df_copy

extended_dataframes_copy = [interpolate_below_value_series(dataframe, column_name=columnName, min_value=0.5) for dataframe in extended_dataframes_copy]
visualize_combined_column(extended_dataframes_copy, columnName)

# %%
