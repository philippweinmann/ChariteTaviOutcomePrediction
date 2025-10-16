import pandas as pd
import numpy as np
import logging
from tavidifficulty.preprocessing.preprocessing_utils import train_apply_transformation, Winsorizer
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

def drop_columns_without_targets(metadata_df, targets_df):
    # 1. merge both
    merged_df = pd.merge(metadata_df, targets_df, on='patient_id', how='right')

    # let's drop the rows without results
    metadata_df = metadata_df[metadata_df.index.isin(merged_df.index)]
    logging.info(f"removed {len(metadata_df) - len(merged_df)} metadata columns without targets")
    
    return metadata_df

def impute_columns(metadata_df):
    '''
    1. gender: random
    2. age: mean
    3. preop_log_euroscore: mean
    4. preol_euroscore_ii: mean
    5. preop_sts_prom: mean
    6. preop_sts_morb or mort: mean
    7. prosthesis_size: biggest category
    '''
    # 1. column: "gender"
    # random imputation
    metadata_df.loc[:,"gender"] = metadata_df["gender"].apply(lambda age:age if pd.notna(age) else np.random.randint(0,2))

    # 2-6. mean imputation columns
    mean_imp_columns = ["age", "preop_log_euroscore", "preop_euroscore_ii", "preop_sts_prom", "preop_sts_morb_or_mort"]
    for mean_imp_column in mean_imp_columns:
        metadata_df.loc[:, mean_imp_column] = metadata_df.loc[:, mean_imp_column].fillna(metadata_df.loc[:, mean_imp_column].median())

    # 7. prothesis size: biggest category
    metadata_df.loc[:,"prosthesis_size"] = metadata_df["prosthesis_size"].fillna(metadata_df["prosthesis_size"].value_counts().index[0])

    return metadata_df

def fix_metadata(metadata_input_df, targets_df):
    # 1. filtering
    metadata_input_df = drop_columns_without_targets(metadata_df=metadata_input_df, targets_df=targets_df)

    # 2. imputation
    metadata_input_df = impute_columns(metadata_input_df)
    
    return metadata_input_df


def normalize_metadata(training_dataframes, non_training_dataframes):
    logging.info("this function assumes, that the data has been fixed. See function (fix_metadata)")

    # PART 1: Clipping
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=Winsorizer(lower=0, upper=0.97), columnName="preop_log_euroscore")
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=Winsorizer(lower=0, upper=0.96), columnName="preop_euroscore_ii")
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=Winsorizer(lower=0, upper=0.96), columnName="preop_sts_prom")
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=Winsorizer(lower=0, upper=0.96), columnName="preop_sts_morb_or_mort")

    # PART 2: Normalization
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=StandardScaler(), columnName="age")
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=StandardScaler(), columnName="preop_log_euroscore")
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=StandardScaler(), columnName="preop_euroscore_ii")
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=MaxAbsScaler(), columnName="preop_sts_prom")
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=MaxAbsScaler(), columnName="preop_sts_morb_or_mort")
    training_dataframes, non_training_dataframes = train_apply_transformation(training_dataframes, non_training_dataframes, scaler=MaxAbsScaler(), columnName="prosthesis_size")

    return training_dataframes, non_training_dataframes