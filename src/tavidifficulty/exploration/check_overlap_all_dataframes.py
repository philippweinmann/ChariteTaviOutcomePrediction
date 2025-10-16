# %%
%load_ext autoreload
%autoreload 2

import pandas as pd
from tavidifficulty.utils import create_pdf_from_dataframe
from IPython.display import display
from tavidifficulty.data.parse_targets import parse_old_targets_file
from tavidifficulty.data.parse_results_files import parse_results_files_to_dfs
from tavidifficulty.data.parse_gender_file import parse_gender_file
from tavidifficulty.data.parse_nifties import parse_nifties
from tavidifficulty.exploration.exploration_utils import check_overlap
# %%
niftie_df = parse_nifties()
results_files_df = parse_results_files_to_dfs() # removes incomplete files.
gender_df = parse_gender_file()
targets_df = parse_old_targets_file()

all_dataframes = [results_files_df, niftie_df, gender_df, targets_df]
all_dataframe_titles = ["results excel files", "nifty images", "gender", "targets_file"]
# %%
overlap_all_df = check_overlap(all_dataframes, all_dataframe_titles)
display(overlap_all_df)
# %%
create_pdf_from_dataframe(overlap_all_df, pdf_filename="overlap_overview.pdf")
# %%
# Johanna asked me for only nifty and result excel files
j_dfs = [results_files_df, niftie_df]
j_titles = ["results excel files", "nifty images"]
j_overlap = check_overlap(j_dfs, j_titles, asc = True)
display(j_overlap)
# %%
create_pdf_from_dataframe(j_overlap, pdf_filename="results_nifty_overlap.pdf")
# %%
only_nifties_df = j_overlap[j_overlap["present in"].apply(lambda x: x == ['nifty images'])]
print(len(only_nifties_df))
print(only_nifties_df.index)
# %%
