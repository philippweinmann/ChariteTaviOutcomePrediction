import pandas as pd
from tavidifficulty.config import p_id_column_name
import matplotlib.pyplot as plt
import textwrap

def check_overlap(dataframes, dataframe_titles, asc=False):
    dataframes_copy = [df.copy() for df in dataframes]

    for dataframe, title in zip(dataframes_copy, dataframe_titles):
        dataframe["present in"] = title

    for i, dataframe in enumerate(dataframes_copy):
        dataframes_copy[i] = dataframe[["present in"]]
        dataframes_copy[i].index = dataframes_copy[i].index.astype(str)

    combined_df = pd.concat(dataframes_copy)

    aggregated_df = (
        combined_df.groupby(combined_df.index)["present in"].agg(list).reset_index()
    )
    aggregated_df = aggregated_df.rename(columns={"index": p_id_column_name})
    aggregated_df = aggregated_df.sort_values(
        by="present in", 
        key=lambda col: col.apply(len), 
        ascending=asc
    )

    aggregated_df.set_index(p_id_column_name, inplace=True)
    return aggregated_df

def visualize_combined_column(dataframes, column_name):
    combined_column = pd.concat([df[[column_name]] for df in dataframes])

    plt.hist(combined_column, bins=50)
    plt.title(f"combined over all patients: {column_name} histogram")
    plt.show()

def compare_visualize_combined_column(dataframe_original, dataframe_modified, column_name):
    combined_column_original = pd.concat([df[[column_name]] for df in dataframe_original])
    combined_column_transformed = pd.concat([df[[column_name]] for df in dataframe_modified])

    fig, axs = plt.subplots(1,2)
    axs[0].hist(combined_column_original, bins=50)
    axs[1].hist(combined_column_transformed, bins=50)

    axs[0].title.set_text("original")
    axs[1].title.set_text("transformed")

    fig.suptitle(f"combined over all patients: {column_name} histogram")
    plt.show()

    return fig

def compare_visualize_combined_columns(dataframe_originals, dataframe_modifieds, column_names):
    combined_column_originals = []
    combined_column_transformeds = []

    for column_name in column_names:
        combined_column_originals.append(pd.concat([df[[column_name]] for df in dataframe_originals]))
        combined_column_transformeds.append(pd.concat([df[[column_name]] for df in dataframe_modifieds]))
    
    n = len(column_names)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n), sharey='row')

    axes[0, 0].set_title("Original", fontsize=28, fontweight='bold')
    axes[0, 1].set_title("Preprocessed", fontsize=28, fontweight='bold')

    for idx, column_name in enumerate(column_names):
        ax_left = axes[idx, 0]
        ax_right = axes[idx, 1]

        # Left plot: Original distribution
        ax_left.hist(combined_column_originals[idx], bins=50, color='skyblue')
        wrapped_label = '\n'.join(textwrap.wrap(column_name, width=15))
        ax_left.set_ylabel(wrapped_label, fontsize=20, rotation=0, 
                          labelpad=120, va='center', fontweight='bold')
        
        # Right plot: Preprocessed distribution
        ax_right.hist(combined_column_transformeds[idx], bins=50, color='lightgreen')
            
        ax_left.tick_params(axis='both', which='major', labelsize=16)
        ax_right.tick_params(axis='both', which='major', labelsize=16)
    

     # Adjust layout with extra padding
    plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=1.0)
    fig.subplots_adjust(top=0.95, left=0.15)  # Extra space for labels
    plt.show()

