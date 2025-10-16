# %%
import pandas as pd
from datetime import datetime
import re
from tavidifficulty.data.resource_config import results_file_without_timestamp, important_sheet_name
import logging
# %%
def get_excel_sheet_df(excel_fp, sheet_name=important_sheet_name):
    excel_file = pd.ExcelFile(excel_fp)
    extracted_sheet_df = excel_file.parse(sheet_name)
    return extracted_sheet_df
    
# %%
def get_excel_sheet_selection(excel_fp, sheet_name, skiprows, cols, nrows=None):
    excel_file = pd.ExcelFile(excel_fp)
    selection_df = pd.read_excel(excel_file, sheet_name=sheet_name, skiprows=skiprows, usecols=cols, nrows=nrows)

    return selection_df

# %%
def convert_timestamp(excel_fp):
    # Regular expression to capture the digits between "@@_" and "_results.xlsx"
    pattern = r'@{0,2}_(\d+)_results\.xlsx'
    match = re.search(pattern, excel_fp)

    if match:
        timestamp_str = match.group(1)
        timestamp = int(timestamp_str)
        # Convert the timestamp to a human-readable datetime
        dt = datetime.fromtimestamp(timestamp)

        return dt
    else:
        raise ValueError(f"No timestamp found in the filepath {excel_fp}.")


def get_most_recent_file(excel_fps):
    # print(f"excel filepaths in getmostrecentfile: {excel_fps}")

    if results_file_without_timestamp in excel_fps:
        excel_fps.remove(results_file_without_timestamp)
        logging.warning(f"removed problematic file without timestamp: {results_file_without_timestamp}")

    # Create a list of tuples (datetime, filepath)
    dt_fp_pairs = [(convert_timestamp(fp), fp) for fp in excel_fps]

    # Get the tuple with the maximum datetime
    latest_datetime, latest_filepath = max(dt_fp_pairs, key=lambda pair: pair[0])
    return latest_filepath, latest_datetime
# %%
