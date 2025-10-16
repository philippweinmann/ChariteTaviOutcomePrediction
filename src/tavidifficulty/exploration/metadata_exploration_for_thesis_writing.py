# %%
%load_ext autoreload
%autoreload 2
import pandas as pd
from tavidifficulty.data.parse_metadata import parse_metadata
from tavidifficulty.data.parse_targets import get_targets_dataframe
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

input_df, output_df = parse_metadata()
unmodified_input_df = input_df.copy()
display(input_df.head().style.set_caption("input metadata"))
display(output_df.head().style.set_caption("output metadata"))
# %%
complication_column_names = ["sev_proc_complic", "aortic_annulus_rupture", "aortic_dissection", "lv_perforation", "perif_access_complications___1", "proc_bleeding___1"]
complications_df = output_df[complication_column_names]
display(complications_df)

print(f"total amount of patients: {len(complications_df)}\n")

for column in complications_df.columns:
    col = complications_df[column]
    amt_confirmed_complications = col.sum()
    print(f"column: {column} sum: {amt_confirmed_complications}")
# %%
print(input_df.columns)
print(input_df.describe())
# %%
# let's put the input_df metadata into a proper table
# we should have the following information
# amount of patients: 108
# amount of columns: 

for column in input_df.columns:
    print(f"column name: {column}, amt of nas: {input_df[column].isna().sum()}")
# %%
targets_df = get_targets_dataframe(old_data=True, new_data=False)
display(targets_df)
# %%
count_values = targets_df["classification_optimal"].value_counts()
print(count_values)
# %%
count_values.plot(kind='bar')

new_labels = ["No major procedure\ncomplication", "Major procedure\ncomplication"]

plt.xticks(ticks=[0,1], labels=new_labels, rotation=0, ha='center')
plt.title("Label distribution")
plt.xlabel("Tavi procedure outcome classification")
plt.show()
# %%
# let's merge both the targets and the input_df on patient_id
display(input_df)
display(targets_df)

# %%
merged_df = pd.merge(input_df, targets_df, left_index=True, right_index=True, how='inner')
display(merged_df)
# %%
male_df = merged_df[merged_df["gender"] == 1.0]
female_df = merged_df[merged_df["gender"] == 0.0]
# %%
def plot_label_distribution(merged_df):
    merged_df_copy = merged_df.copy()

    label_map = {True: "No major procedure\ncomplication", False: "Major procedure\ncomplication"}
    merged_df_copy['classification_optimal'] = merged_df_copy['classification_optimal'].map(label_map)

    # Separate data by gender
    male_df = merged_df_copy[merged_df_copy["gender"] == 1.0]
    female_df = merged_df_copy[merged_df_copy["gender"] == 0.0]
    
    # Count occurrences of each label
    count_values_male = male_df["classification_optimal"].value_counts()
    count_values_female = female_df["classification_optimal"].value_counts()

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5), sharey=True)
    fig.suptitle("Label Distribution by Gender", fontsize=14)

    # Plot male distribution
    count_values_male.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
    ax1.set_title('Male (Gender=1.0)')
    ax1.set_xlabel('Tavi procedure outcome classification')
    ax1.tick_params(axis='x', rotation=0)

    # Plot female distribution
    count_values_female.plot(kind='bar', ax=ax2, color='lightcoral', edgecolor='black')
    ax2.set_title('Female (Gender=0.0)')
    ax2.set_xlabel('Tavi procedure outcome classification')
    ax2.tick_params(axis='x', rotation=0)

    # Adjust layout and display
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.show()

plot_label_distribution(merged_df)
# %%
import matplotlib.pyplot as plt
import pandas as pd # Assuming you are using pandas for the DataFrame

def plot_label_distribution(merged_df):
    merged_df_copy = merged_df.copy()

    label_map = {True: "No major procedure\ncomplication", False: "Major procedure\ncomplication"}
    merged_df_copy['classification_optimal'] = merged_df_copy['classification_optimal'].map(label_map)

    # Separate data by gender
    male_df = merged_df_copy[merged_df_copy["gender"] == 1.0]
    female_df = merged_df_copy[merged_df_copy["gender"] == 0.0]
    
    # Count occurrences of each label
    count_values_male = male_df["classification_optimal"].value_counts()
    count_values_female = female_df["classification_optimal"].value_counts()

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6), sharey=True) # Increased figure height slightly
    # --- FONTSIZE INCREASED ---
    fig.suptitle("Label Distribution by Gender", fontsize=24) 

    # Plot male distribution
    count_values_male.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
    # --- FONTSIZE INCREASED ---
    ax1.set_title('Male (Gender=1.0)', fontsize=20)
    ax1.set_xlabel('Tavi procedure outcome classification', fontsize=14)
    # --- FONTSIZE INCREASED for tick labels ---
    ax1.tick_params(axis='x', rotation=0, labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)


    # Plot female distribution
    count_values_female.plot(kind='bar', ax=ax2, color='lightcoral', edgecolor='black')
    # --- FONTSIZE INCREASED ---
    ax2.set_title('Female (Gender=0.0)', fontsize=20)
    ax2.set_xlabel('Tavi procedure outcome classification', fontsize=14)
    # --- FONTSIZE INCREASED for tick labels ---
    ax2.tick_params(axis='x', rotation=0, labelsize=16)


    # Adjust layout and display
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make room for larger suptitle
    plt.show()

plot_label_distribution(merged_df)
# %%
