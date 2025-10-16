# %%
from pathlib import Path
import os
import pandas as pd
from tavidifficulty.utils import create_pdf_from_dataframe, extract_patient_id
from tavidifficulty.config import p_id_column_name
import pickle
# %%
parent_folder = Path("/srv/data/TAVIDifficulty/charite_data/tavi_tricuspid_ct_wds_31_03/TAVI_Tricuspid_CT_WDs")
results_pickle_fn = "/srv/data/TAVIDifficulty/tavidifficulty/src/tavidifficulty/saved_dataframes/results_df.pkl"
# %%
# Initialize list to hold subfolder information
data = []

# Loop over each item in the parent folder
for item in os.listdir(parent_folder):
    subfolder_path = os.path.join(parent_folder, item)
    # Check if the item is a directory (subfolder)
    if os.path.isdir(subfolder_path):
        # List all items in the subfolder
        files = os.listdir(subfolder_path)
        folder_empty = len(files) == 0  # Check if subfolder is empty
        
        # Count Excel files with "result" in their filename (case-insensitive)
        count_results = 0
        for f in files:
            file_path = os.path.join(subfolder_path, f)
            if os.path.isfile(file_path):
                # Check if the file has an Excel extension and contains "result" in the name
                if f.lower().endswith(('.xls', '.xlsx', '.xlsm')) and 'result' in f.lower():
                    count_results += 1
        
        # Determine if at least one results file is present
        results_file_present = count_results > 0
        
        # Append the information for this subfolder to the data list
        data.append({
            p_id_column_name: extract_patient_id(item),
            'results file present': results_file_present,
            'how many are present': count_results,
            'folder completely empty': folder_empty
        })

# Create the DataFrame from the collected data
df = pd.DataFrame(data)
# %%
df = df.sort_values(by=["how many are present"])
df["issue detected"] = df["how many are present"] == 0
# %%
with open(results_pickle_fn, mode="wb") as f:
    pickle.dump(df, f)
# %%
create_pdf_from_dataframe(df, "results_excel_issues_new.pdf")
# %%
