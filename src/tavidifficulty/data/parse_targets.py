# %%
import pandas as pd
from tavidifficulty.config import old_johanna_targets_excel_fp, new_isaac_targets_excel_fp, p_id_column_name
from IPython.display import display
import logging

# %%
def parse_target_file(targets_excel_fp):
    # common method for both excel files
    target_df = pd.read_excel(targets_excel_fp)
    # let's strip all column names and put lower case
    target_df.columns = target_df.columns.str.strip()
    target_df.columns = target_df.columns.str.lower()
    target_df = target_df.rename(columns={"pid": p_id_column_name})
    # let's drop any duplicates
    target_df.drop_duplicates(subset=[p_id_column_name], inplace=True)
    # let's set the index to PID
    target_df.set_index(keys=[p_id_column_name], inplace=True)
    target_df.index = target_df.index.astype(int)

    # let's prepare the classification column for merge
    target_df['classification'] = target_df['classification'].str.strip()
    target_df['classification'] = target_df['classification'].str.lower()

    logging.info(f"Number of unique patients: {len(target_df)}")
    return target_df

def apply_modifications(targets_df):
    # let's write another column, with the type of the device
    targets_df['device'] = targets_df['classification'].apply(lambda x: x[0:2])

    # let's drop the first 3 chars from the classification column
    targets_df['classification'] = targets_df['classification'].apply(lambda x: x[3:
    ])

    # let's OHE the classification column
    targets_df = pd.get_dummies(targets_df, columns=['classification'], prefix="classification", drop_first=True)

    # let's OHE the device column
    targets_df = pd.get_dummies(targets_df, columns=['device'], prefix="device", drop_first=True)

    return targets_df

def get_merged_target_df():
    old_target_df = parse_target_file(old_johanna_targets_excel_fp)
    new_target_df = parse_target_file(new_isaac_targets_excel_fp)

    # let's merge on index
    merged_df = pd.merge(old_target_df, new_target_df, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))

    # Step 2: Check for conflicts
    col1 = 'classification_1'
    col2 = 'classification_2'
    
    # Define blank values (NaN or empty string)
    blank1 = merged_df[col1].isna() | (merged_df[col1] == '')
    blank2 = merged_df[col2].isna() | (merged_df[col2] == '')
    
    # Non-blank masks
    non_blank1 = ~blank1
    non_blank2 = ~blank2
    
    # Conflict: Both non-blank and values differ
    conflicts = non_blank1 & non_blank2 & (merged_df[col1] != merged_df[col2])
    
    if conflicts.any():
        conflicting_indices = merged_df.index[conflicts].tolist()
        logging.warning(f"Conflicting Classifications at indices: {conflicting_indices}")
        logging.warning(f"dropping them")
        merged_df.drop(labels=conflicting_indices, inplace=True)
    
    # Step 3: Combine columns
    # Use col1 where it's non-blank; otherwise use col2 (which may be blank)
    merged_df['classification'] = merged_df[col1]
    merged_df.loc[blank1, 'classification'] = merged_df.loc[blank1, col2]
    
    # Step 4: Drop temporary columns
    merged_df = merged_df.drop(columns=[col1, col2])

    merged_df = apply_modifications(merged_df)

    return merged_df

def get_targets_dataframe(old_data=False, new_data=False):
    if old_data & new_data:
        return get_merged_target_df()
    
    if old_data:
        target_fp = old_johanna_targets_excel_fp
    elif new_data:
        target_fp = new_isaac_targets_excel_fp
    else:
        raise ValueError("both old_data and new_data cannot be false")
    
    # the process is the the same for old and new data
    targets_df = parse_target_file(target_fp)
    targets_df = apply_modifications(targets_df)
    return targets_df



# target_df = parse_targets()
# target_df.head(n=10)
# %%
# test
# display(parse_targets().loc["2015626"])
# %%
