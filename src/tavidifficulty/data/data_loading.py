# %%
# this file is meant to help you load all of the data to train a model.
# move to different files during refactoring or whatever.
import numpy as np
import logging
# %%
def get_extended_data(patients):
    # returns a list of input and output data of the extended
    # (slice by slice) data.
    
    input_dataframes = []
    targets = []
    metadatas = []

    for patient in patients:
        input_dataframes.append(patient.get_extended_params_without_abdominal_aorta)
        metadatas.append(patient.metadata_series)
        targets.append(patient.optimal)

    return input_dataframes, metadatas, np.array(targets)
# %%
# save_patients_to_pickle()

def pad_if_necessary(padded_length, X):
    # max_length_without_abdominal_aorta = 329
    padded_X = []

    for xs in X:
        # print(f"lenght: {xs.shape[1]}")
        pad_length = padded_length - xs.shape[1]
        padded_xs = np.pad(xs, ((0, 0), (0, pad_length)), mode='constant')
        padded_X.append(padded_xs)

    return np.array(padded_X)
# %%
