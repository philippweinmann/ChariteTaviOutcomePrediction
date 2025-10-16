# %%
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from tavidifficulty.data.resource_config import metadata_csv_fp, id_column_name
# %%
# let's rename the columns for more clarity
rename_dict = {"pid": id_column_name, "sex":"gender", "preop_age":"age", "prothesis_size_1":"prosthesis_size",}

def parse_metadata(metadata_fp = metadata_csv_fp):
    df = pd.read_csv(metadata_fp)

    # let's set the id to the index and more changes for clarification
    df = df.rename(columns=rename_dict)
    df.set_index(keys=[id_column_name], inplace=True)
    df.index = df.index.astype(int)

    # let's drop Unnamed?? wtf is this even
    df = df.drop(columns=['Unnamed: 0'])

    # let's remove column prosthesis_1, Isaac does not know what it means.
    df = df.drop(columns=['prosthesis_1'])

    trad_score_columns = ['preop_log_euroscore',
       'preop_euroscore_ii', 'preop_sts_prom', 'preop_sts_morb_or_mort']

    # columns that could be output:
    output_columns = ['prosthesis_repositioning', 'sev_proc_complic', 'aortic_annulus_rupture', 'aortic_dissection', 'lv_perforation', 'perif_access_complications___1', 'proc_bleeding___1',
       'survival_time']
    
    output_df = df[output_columns].copy()
    input_df = df.drop(columns=output_columns)

    return input_df, output_df