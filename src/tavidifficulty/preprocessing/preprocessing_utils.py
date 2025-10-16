import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.05, upper=0.95):
        self.lower = lower
        self.upper = upper
        
    def fit(self, X, y=None):
        self.lower_bounds_ = X.quantile(self.lower)
        self.upper_bounds_ = X.quantile(self.upper)
        return self
        
    def transform(self, X):
        return X.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1).to_numpy()


def train_apply_transformation(training_dataframes, non_training_dataframes, scaler, columnName):
    # I am proud of that function, makes the code really clean
    combined_train_column = pd.concat([df[[columnName]] for df in training_dataframes])
    scaler.fit(combined_train_column)

    for dataframe in training_dataframes:
        transformed_col = scaler.transform(dataframe[[columnName]]).ravel()
        dataframe.loc[:,columnName] = transformed_col

    for dataframe in non_training_dataframes:
        transformed_col = scaler.transform(dataframe[[columnName]]).ravel()
        dataframe.loc[:,columnName] = transformed_col

    return training_dataframes, non_training_dataframes