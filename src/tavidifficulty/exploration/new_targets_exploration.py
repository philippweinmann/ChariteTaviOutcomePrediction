# %%
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from tavidifficulty.data.resource_config import metadata_csv_fp, id_column_name
from tavidifficulty.data.parse_targets import parse_old_targets_file
# %%
# let's rename the columns for more clarity
rename_dict = {"pid": id_column_name, "sex":"gender", "preop_age":"age", "prothesis_size_1":"prosthesis_size_1",}

def parse_new_targets(target_fp = metadata_csv_fp):
    df = pd.read_csv(target_fp)
    print(df.columns)

    # let's set the id to the index and more changes for clarification
    df = df.rename(columns=rename_dict)
    df.set_index(keys=[id_column_name], inplace=True)
    df.index = df.index.astype(int).astype(str)

    # let's drop Unnamed?? wtf is this even
    df = df.drop(columns=['Unnamed: 0'])

    # let's drop the euroscores, maybe I'll use them later?
    trad_score_columns = ['preop_log_euroscore',
       'preop_euroscore_ii', 'preop_sts_prom', 'preop_sts_morb_or_mort']
    df = df.drop(columns=trad_score_columns)

    # columns that could be output:
    output_columns = ['prosthesis_repositioning', 'sev_proc_complic', 'aortic_annulus_rupture', 'aortic_dissection', 'lv_perforation', 'perif_access_complications___1', 'proc_bleeding___1',
       'survival_time']
    
    output_df = df[output_columns].copy()
    input_df = df.drop(columns=output_columns)

    return input_df, output_df
# %%
input_df, output_df = parse_new_targets()

# %% 
# let's join with the old target column
old_targets = parse_old_targets_file()
output_df = input_df.join(output_df, how='outer').join(old_targets, how='outer')
display(output_df)
# %%
# we will now look at one columns at a time. There are multiple goals.
# 1. Understand what that data ist
# 2. Combine it into one columns to continue iteratively working on data
# 3. I doubt it, but maybe we can divide it up into multiple categories, let's see.

# first column: prosthesis repostitioning
# not possible for all valve designs.
def quick_values_count(series):
    display(series)
    print(f"amount of Nan values: {series.isna().sum()}")

    values_count = series.value_counts(dropna=False)
    values_count.plot(kind='bar')

quick_values_count(output_df["prosthesis_repositioning"])
# conclusion: Not a great value, let's wait for isaacs data.
# %%
# next column: sev_proc_complic
quick_values_count(output_df["sev_proc_complic"])

# not great either
# %%
# next column: aortic_annulus_rupture
# this one is just empty? Let's ask Isaac about it
quick_values_count(output_df["aortic_annulus_rupture"])
# %%
# next column: aortic_dissection
# same here
quick_values_count(output_df["aortic_dissection"])
# %%
quick_values_count(output_df["lv_perforation"])
# %%
quick_values_count(output_df["perif_access_complications___1"])
# okay there's data here.
# %%
# next one: proc_bleeding___1
quick_values_count(output_df["proc_bleeding___1"])
# %%
# next: survival_time
# let's check nans
print(output_df["survival_time"].isna().sum()) # 12
plt.hist(output_df["survival_time"], bins=20)

# %%
len(output_df)
# %%
# We've learnt that some issues seem to appear very often. just because they're 1s does not mean, that the
# operation didn't go well.

# let's sanity check the data.
# let's check the rows sev_proc_complic and Classification_optimal
sev_complic_df = output_df[output_df['sev_proc_complic'] == 1.0]
sev_complic_df.head()

# okay holds up, but its only 3
# %%
sev_complic_df = output_df[output_df['aortic_annulus_rupture'] == 1.0]
sev_complic_df.head()
# holds up as well, its only 1
# %%
sev_complic_df = output_df[output_df['prosthesis_repositioning'] == 1.0]
sev_complic_df.head()
# one is different?? But seems to still be OK
# %%
bad_stuff_happened_cols = ["sev_proc_complic", "aortic_annulus_rupture", "aortic_dissection", "lv_perforation"]

for bad_stuff_happened_col in bad_stuff_happened_cols:
    print(bad_stuff_happened_col)
    comp_df = output_df[output_df[bad_stuff_happened_col] == 1.0]
    display(comp_df[[bad_stuff_happened_col, "classification_optimal"]])
# %%
# I suppose it is still okay to use the column "classification optimal". I'll question isaac about it, but it seems okay.

