# %%
import pandas as pd
import pickle
from tavidifficulty.config import p_id_column_name
import pandas as pd
from IPython.display import display
from tavidifficulty.data.parse_targets import parse_old_targets_file
# %%

# %%
dataframe_fp = "/srv/data/TAVIDifficulty/tavidifficulty/src/tavidifficulty/saved_dataframes"
results_pkl_fp = dataframe_fp + "/results_df.pkl"
with open(results_pkl_fp, mode="rb") as f:
    results_df = pickle.load(f)

nifty_pkl_fp = dataframe_fp + "/nifty_df.pkl"
with open(nifty_pkl_fp, mode="rb") as f:
    nifty_df = pickle.load(f)

targets_df = parse_old_targets_file()
# %%
results_df.head()

# %%
display(nifty_df)
# %%
def merge_df(df1, df2):
    merged_df = pd.merge(df1, df2, on=p_id_column_name, how='inner')

    # Count the total rows in the merged dataframe
    total_rows = merged_df.shape[0]

    # Count rows with issues and without issues.
    # This example assumes that the "issue detected" column contains boolean values.
    rows_with_issue = merged_df[merged_df['issue detected']].shape[0]
    rows_without_issue = merged_df[~merged_df['issue detected']].shape[0]

    print("Total merged rows:", total_rows)
    print("Rows with issues:", rows_with_issue)
    print("Rows without issues:", rows_without_issue)

    return merged_df

# %%
print("checking targets overlap with results excel files")
merged_df = merge_df(results_df, targets_df)
print(len(merged_df))
# %%
display(merged_df)
# %%
print(f" amount of patients with results files: {len(results_df)}")
print(f"amt of patients with targets: {len(targets_df)}")

print(f"difference: {len(results_df) - len(targets_df)}")
# %%
display(merged_df)
# %%
