# %%
import logging
from tavidifficulty.preprocessing.preprocessing_utils import Winsorizer, train_apply_transformation
import numpy as np

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

# 'Min Diameter[mm]'
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
    df_copy.loc[zero_mask, column_name] = np.nan
    # Perform linear interpolation
    df_copy[column_name] = df_copy[column_name].interpolate(method='linear')

    return df_copy

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
# %%