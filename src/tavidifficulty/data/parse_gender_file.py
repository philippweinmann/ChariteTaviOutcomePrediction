from tavidifficulty.data.resource_config import gender_fp
import pandas as pd
from tavidifficulty.config import p_id_column_name


def parse_gender_file(gender_fp=gender_fp):
    df = pd.read_csv(gender_fp)
    df = df.rename(columns={"case_id": p_id_column_name})
    df = df.drop(columns=["Unnamed: 0"])
    df.index = df[p_id_column_name].astype(int)
    df = df.drop(columns=[p_id_column_name])

    return df