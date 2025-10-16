# %%
%load_ext autoreload
%autoreload 2
import pandas as pd
from tavidifficulty.data.parse_metadata import parse_metadata
from tavidifficulty.data.parse_targets import get_targets_dataframe
from tavidifficulty.preprocessing.preprocess_metadata import fix_metadata, normalize_metadata
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
import seaborn as sns
import copy
# %%
input_df, output_df = parse_metadata()
unmodified_input_df = input_df.copy()
display(input_df.head().style.set_caption("input metadata"))
display(output_df.head().style.set_caption("output metadata"))
# %%
columns_to_ignore = ["prosthesis_type"]
input_df.drop(columns=columns_to_ignore, inplace=True)

# %%
for columnName in input_df.columns:
    col = input_df[columnName]
    plt.title(columnName)
    plt.hist(col, bins=50)
    plt.show()

# %%
'''
Let's think about the imputation methods we might be able to use.
1. gender: just do it randomly
2. age: do mean age. Yes you could do mean age per gender group, but no need.
3. preop_log_euroscore: what is up with the 0s? investigate further.
4. preol_euroscore_ii: what is up with the 0s? looks weird.
5. preop_sts_prom: same thing?
6. preop_sts_morb or mort: Is that the same data as preop sts prom? check correlation
7. prothesis size: don't take mean, put it into the biggest category.
'''
# %%
# 5. preop_sts_prom: same thing?

# let's check the correlation between columns. Maybe we can remove one column alltogether.
corr = input_df.corr()
sns.heatmap(corr, cmap="coolwarm", annot=True)
plt.show()
# nope we keep all columns, the correlations aren't strong enough between the two sts columns
# %%
# 3. preop_log_euroscore: what is up with the 0s? investigate further.
# 4. preol_euroscore_ii: what is up with the 0s? looks weird.
# with pd.option_context("display.max_rows", None):
#     display(input_df.style.set_caption("input metadata"))
# let's count the amount of zeroes per column:
for columnName in input_df.columns:
    amt_zeroes = (input_df[columnName] == 0).sum()
    print(f"column: {columnName} has {amt_zeroes} zero values")

# okay the values are just very small, they're not actually zero.
# we can proceed.
# %%
# let's see which columns even require imputations
for column in input_df.columns:
    print(f"column name: {column}, amt of nas: {input_df[column].isna().sum()}")

# okay... all of them
# %%
'''
let's update out imputation strategies:
1. gender: random
2. age: mean
3. preop_log_euroscore: mean
4. preol_euroscore_ii: mean
5. preop_sts_prom: mean
6. preop_sts_morb or mort: mean
7. prothesis size: biggest category
'''
# %%
targets_df = get_targets_dataframe(old_data=True, new_data=False)
org_data = copy.deepcopy(input_df)

metadata_df = fix_metadata(input_df, targets_df)
# %%
# let's compare data before preprocessing, and after preprocessing.
print("original data:")
print(org_data.isna().sum())

print("imputated data: ")
print(metadata_df.isna().sum())
# %%
def vis_metadata_transformation(org_col, transformed_col, title):
    amt_bins=30
    fig, axs = plt.subplots(1,2)

    axs[0].hist(org_col, bins=amt_bins)
    axs[1].hist(transformed_col, bins=amt_bins)

    axs[0].set_ylim([0, 80])
    axs[1].set_ylim([0, 80])

    fig.suptitle(title)
    plt.show()

for col in org_data.columns:
    # the transformation includes dropping columns without targets, therefore we have less values overall
    vis_metadata_transformation(org_data[col], metadata_df[col], "transformation visualization of col: " + col)

# %%
for col in org_data.columns:
    plt.hist(metadata_df[col], bins=40)
    plt.title(col)
    plt.show()
# %%
print(org_data.columns)

# %%
training_dataframe, non_training_dataframe = metadata_df.iloc[1:], metadata_df.iloc[:1]
[training_dataframe], [non_training_dataframe] = normalize_metadata([training_dataframe], [non_training_dataframe])

for col in training_dataframe.columns:
    vis_metadata_transformation(org_data[col], training_dataframe[col], "transformation visualization of col: " + col)
# %%
