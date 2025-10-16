# %%
# %%
%load_ext autoreload
%autoreload 2
from tavidifficulty.data.data_loading import get_extended_data, pad_if_necessary
from tavidifficulty.preprocessing.preprocessing_extended_data import remove_unfinished_sections, interpolate_zero_series
from tavidifficulty.data.parse_results_files import parse_all_results_and_targets_files_to_patients
from tavidifficulty.config import max_length_without_abdominal_aorta
import copy
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from tavidifficulty.exploration.exploration_utils import apply_previous_preprocessing_steps
# %%
patients = parse_all_results_and_targets_files_to_patients(remove_incomplete=True, remove_without_targets=True)
extended_dataframes = [patient.get_extended_params_without_abdominal_aorta for patient in patients]
# %%
# fixing steps to apply:
# 1. remove unfinished sections

# transformation steps to apply:
# 1. MaxAbsScale Distance from valve
# 2. Clip values at 3 and 97th percentile

# we copy these functions to get a snapshot of how they looked at the time of exploration
from sklearn.preprocessing import MaxAbsScaler
from tavidifficulty.preprocessing.preprocessing_utils import Winsorizer

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
    dataframes, _ = train_apply_scaler(dataframes, [], scaler=Winsorizer(lower=0.03, upper=0.97), columnName="Crosssection Area[mm2]")

    return dataframes

df_pp = apply_previous_preprocessing_functions(extended_dataframes)
original_df_pp = copy.deepcopy(df_pp)
# %%
current_column_name = 'Area Diameter[mm]'

def plot_hist(dataframe):
    plt.hist(dataframe[current_column_name], bins=50)
    plt.show()

def plot_all_hists(dataframes):
    for i, df in enumerate(dataframes):
        print(f"index i: {i}")
        plt.hist(df[current_column_name], bins=50)
        plt.show()

def plot_all_hists_compare(df, df_original):
    for i, (df1, df2) in enumerate(zip(df, df_original)):
        fig, axs = plt.subplots(nrows=1, ncols=2)
        axs[0].hist(df1[current_column_name], bins=50)
        axs[1].hist(df2[current_column_name], bins=50)
        plt.suptitle(i)
        plt.show()

def plot_random_hists(dataframes):
    rand_indexes = np.random.choice(np.arange(len(dataframes)), size=10, replace=False)
    for i in range(10):
        curr_df = dataframes[rand_indexes[i]].loc[:,current_column_name]
        plt.hist(curr_df, bins=50)
        plt.show()

# plot_all_hists(df_pp)
# %%
# why can't things be simple? There is an issue around 25. Let#s investigate.
plot_hist(df_pp[103])
# %%
pd.set_option('display.max_rows', None)

# display(df_pp[103])
# hmm seems fine to me, I just needed more bins
# %%
# okay the rest is pretty straight forward. We fix the 0s with linear interpolation and winsorizing
# fix the 0s with linear interpolation
df_pp = [interpolate_zero_series(dataframe, column_name="Area Diameter[mm]") for dataframe in df_pp]
plot_all_hists_compare(df_pp, original_df_pp)
# %%
display(df_pp[76])
# %%
plot_hist(df_pp[76])
# %%
print(df_pp[76][current_column_name].min())
# %%
# there are still some zeroes, strange, let's hope clipping will get rid of them.
display(df_pp[7])
# %%
display(df_pp[12][current_column_name])
display(original_df_pp[12][current_column_name])
# %%
