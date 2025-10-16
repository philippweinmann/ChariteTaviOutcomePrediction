# the nifty images can obviously not be parsed, however we can extract some metadata.
# %%
import nibabel as nib
import glob
import numpy as np
import pandas as pd
from IPython.display import display
from pathlib import Path
from tavidifficulty.utils import extract_patient_id
from tavidifficulty.config import p_id_column_name
from tavidifficulty.data.resource_config import nifty_parent_folder_fp
import multiprocessing
# %%
def get_dims(nifty_fp):
    file = nib.load(nifty_fp)
    data = file.get_fdata()

    shape = data.shape
    print(f"shape for scan at {nifty_fp}: {shape}")

    return shape

def parse_nifty_file(nifty_fp):
        file = nib.load(nifty_fp)
        data = file.get_fdata()

        filename = extract_patient_id(Path(nifty_fp).name)
        shape = data.shape

        mean = np.mean(data)
        max = np.max(data)
        min = np.min(data)
        std = np.std(data)

        new_row = {p_id_column_name: filename, "filepath": nifty_fp, "shape": shape, "mean": mean, "min": min, "max": max, "std": std}
        return new_row

def parse_nifties(nifty_parent_folder_fp = nifty_parent_folder_fp):
    print("parsing nifty files. This could take a minute")
    nifty_files = glob.glob(str(nifty_parent_folder_fp / "*.nii"))

    data_summary = []

    with multiprocessing.Pool(processes=48) as p:
         data_summary = p.map(parse_nifty_file, nifty_files)

    data_summary_df = pd.DataFrame(data_summary)
    data_summary_df.set_index(p_id_column_name, inplace=True)

    return data_summary_df