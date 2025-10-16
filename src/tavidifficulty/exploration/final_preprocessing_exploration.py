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
import logging
from tavidifficulty.preprocessing.preprocessing_utils import Winsorizer
from tavidifficulty.exploration.exploration_utils import visualize_combined_column, compare_visualize_combined_columns, compare_visualize_combined_column
# %%
# patients = parse_all_results_and_targets_files_to_patients(use_new_targets=False, use_old_targets=True, remove_incomplete=True, remove_without_targets=True)
# extended_dataframes = [patient.get_extended_params_without_abdominal_aorta for patient in patients]

# print(extended_dataframes[0].columns)
# print(extended_dataframes[0].dtypes)
# %%
# for column in extended_dataframes[0].columns:
#     visualize_combined_column(extended_dataframes, column)
# %%
# let's apply the current preprocessing functions to see the impact
from tavidifficulty.preprocessing.preprocessing_config import MIN_DISTANCE_TO_VALVE, MAX_DISTANCE_TO_VALVE, MIN_CROSSSECTION, MAX_CROSSSECTION
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
    df_copy.loc[zero_mask, column_name] = np.nan
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
# # 1. fixing preprocessing step: remove unfinished sections

# extended_dataframes_copy = copy.deepcopy(extended_dataframes)
# extended_dataframes_removed_unfinished_sections = [remove_unfinished_sections(dataframe) for dataframe in extended_dataframes_copy]

# for column in extended_dataframes[0].columns:
#     compare_visualize_combined_column(dataframe_original=extended_dataframes_copy, dataframes_modified=extended_dataframes_removed_unfinished_sections, column_name=column)

# # notice that we fixed issues in columns "distance from valve" and "distance from arch center", .
# # %%
# # 2. fixing preprocessing step: interpolate zero series
# extended_dataframes_removed_unfinished_sections_copy = copy.deepcopy(extended_dataframes_removed_unfinished_sections)
# extended_dataframes_interpolate_zero_series = [interpolate_zero_series(dataframe, column_name="Crosssection Area[mm2]") for dataframe in extended_dataframes_removed_unfinished_sections_copy]

# compare_visualize_combined_column(dataframe_original=extended_dataframes_removed_unfinished_sections_copy, dataframes_modified=extended_dataframes_interpolate_zero_series, column_name="Crosssection Area[mm2]")

# # %%
# # 3. fixing preprocessing step: interpolate column area
# extended_dataframes_interpolate_zero_series_copy = copy.deepcopy(extended_dataframes_interpolate_zero_series)
# extended_dataframes_interpolate_area_diameter = [interpolate_zero_series(dataframe, column_name="Area Diameter[mm]") for dataframe in extended_dataframes_interpolate_zero_series_copy]

# compare_visualize_combined_column(dataframe_original=extended_dataframes_interpolate_zero_series_copy, dataframes_modified=extended_dataframes_interpolate_area_diameter, column_name="Area Diameter[mm]")
# # %%
# for column in extended_dataframes[0].columns:
#     visualize_combined_column(extended_dataframes_interpolate_area_diameter, column)
# # %%
# # let's ignore fixing step 2 and 3. I will apply more steps for extended_dataframes_removed_unfinished_sections
# # 1. distance from valve
# extended_dataframes_removed_unfinished_sections_copy2 = copy.deepcopy(extended_dataframes_removed_unfinished_sections)
# extended_dataframes_removed_unfinished_sections_copy2, _ = train_apply_transformation(extended_dataframes_removed_unfinished_sections_copy2, [], scaler=MaxAbsScaler(), columnName="Distance from Valve[mm]")

# # 2. cross section
# # clip values
# extended_dataframes_removed_unfinished_sections_copy2, _ = train_apply_transformation(extended_dataframes_removed_unfinished_sections_copy2, [], scaler=Winsorizer(lower=0.02, upper=0.98), columnName="Crosssection Area[mm2]")

# # 3. Area diameter
# # clip values
# extended_dataframes_removed_unfinished_sections_copy2, _ = train_apply_transformation(extended_dataframes_removed_unfinished_sections_copy2, [], scaler=Winsorizer(lower=0.00, upper=0.98), columnName="Area Diameter[mm]")

# # 4. Min Diameter
# # clip values
# extended_dataframes_removed_unfinished_sections_copy2, _ = train_apply_transformation(extended_dataframes_removed_unfinished_sections_copy2, [], scaler=Winsorizer(lower=0.02, upper=0.98), columnName="Min Diameter[mm]")

# # 5. Curvature
# # only clip upper
# extended_dataframes_removed_unfinished_sections_copy2, _ = train_apply_transformation(extended_dataframes_removed_unfinished_sections_copy2, [], scaler=Winsorizer(lower=0.00, upper=0.98), columnName="Curvature[1/mm]")

# # 6. Distance from arch center

# # 7. Distance from Arch plane
# extended_dataframes_removed_unfinished_sections_copy2, _ = train_apply_transformation(extended_dataframes_removed_unfinished_sections_copy2, [], scaler=Winsorizer(lower=0.02, upper=0.98), columnName="Distance from Arch Plane [mm]")

# for column in extended_dataframes[0].columns:
#     compare_visualize_combined_column(dataframe_original=extended_dataframes_removed_unfinished_sections, dataframes_modified=extended_dataframes_removed_unfinished_sections_copy2, column_name=column)
# %%
# CalcCol =  pd.concat([df[["Calcification [ml]"]] for df in extended_dataframes_removed_unfinished_sections_copy2])
# print(CalcCol.quantile(q=0.995))
# %%
patients = parse_all_results_and_targets_files_to_patients(use_new_targets=True, use_old_targets=False, remove_incomplete=True, remove_without_targets=True)
extended_dataframes = [patient.get_extended_params_without_abdominal_aorta for patient in patients]

def fix_columns(dataframes):
    # 1. distance from valve
    dataframes = [remove_unfinished_sections(dataframe) for dataframe in dataframes]

    # 2. cross section
    dataframes = [interpolate_zero_series(dataframe, column_name="Crosssection Area[mm2]") for dataframe in dataframes]

    # 3. column area
    # doesn't work, there might be an issue here
    # function behaves as expected but the models performance becomes significantly worse
    # dataframes = [interpolate_zero_series(dataframe, column_name="Area Diameter[mm]") for dataframe in dataframes]

    # 4. Min Diameter[mm]
    # doesn't work, there might be an issue here
    # function behaves as expected but the models performance becomes significantly worse
    # dataframes = [interpolate_below_value_series(dataframe, column_name='Min Diameter[mm]', min_value=0.5) for dataframe in dataframes]

    return dataframes

def normalize_columns(training_dataframes, non_training_dataframes):
    logging.info("this function assumes, that the data has been fixed. See function (fix_columns)")

    # PART 1: Clipping
    # 1. distance from valve
    # clipping not appropriate

    # 2. cross section
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=Winsorizer(lower=0.02, upper=0.98), columnName="Crosssection Area[mm2]")

    # 3. Area diameter
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=Winsorizer(lower=0.00, upper=0.98), columnName="Area Diameter[mm]")

    # 4. Min Diameter
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=Winsorizer(lower=0.02, upper=0.98), columnName="Min Diameter[mm]")

    # 5. Curvature
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=Winsorizer(lower=0.00, upper=0.99), columnName="Curvature[1/mm]")

    # 6. Distance from arch center

    # 7. Distance from Arch plane
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=Winsorizer(lower=0.02, upper=0.98), columnName="Distance from Arch Plane [mm]")

    # 8. Calcification
    # only upper clipping
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=Winsorizer(lower=0.00, upper=0.995), columnName="Calcification [ml]")

    # PART 2: normalization
    # 1. Distance from valve
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=MaxAbsScaler(), columnName="Distance from Valve[mm]")

    # 2. cross section
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=StandardScaler(), columnName="Crosssection Area[mm2]")

    # 3. Area Diameter
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=StandardScaler(), columnName="Area Diameter[mm]")

    # 4. Min Diameter
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=RobustScaler(), columnName="Min Diameter[mm]")

    # 5. Curvature
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=StandardScaler(), columnName="Curvature[1/mm]")

    # 6. Distance from arch center
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=MaxAbsScaler(), columnName="Distance from Arch Center [mm]")

    # 7. Distance from Arch plane
    # Todo test out if standardscaler is not more appropriate here
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=MaxAbsScaler(), columnName="Distance from Arch Plane [mm]")

    # 8. Calcification
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=MaxAbsScaler(), columnName="Calcification [ml]")

    return training_dataframes, non_training_dataframes

extended_dataframes_og = [patient.get_extended_params_without_abdominal_aorta for patient in patients]
preprocessed_extended_dataframes_og = copy.deepcopy(extended_dataframes_og)

preprocessed_extended_dataframes_og = fix_columns(preprocessed_extended_dataframes_og)
preprocessed_extended_dataframes_og, [] = normalize_columns(preprocessed_extended_dataframes_og, [])
# %%
figs = []

for column in extended_dataframes[0].columns:
    fig = compare_visualize_combined_column(dataframe_original=extended_dataframes_og, dataframe_modified=preprocessed_extended_dataframes_og, column_name=column)
    figs.append(fig)

# %%
# %%
compare_visualize_combined_columns(extended_dataframes_og[0:4], preprocessed_extended_dataframes_og[0:4], extended_dataframes[0].columns[0:4])
# %%
compare_visualize_combined_columns(extended_dataframes_og[4:], preprocessed_extended_dataframes_og[4:], extended_dataframes[0].columns[4:])
# %%
