# %%
import pandas as pd
from IPython.display import display
from tavidifficulty.config import p_id_column_name

sex_fp = "/srv/data/TAVIDifficulty/case_sex.csv"

# %%
def parse_sex_info(sex_excel_fp):
    df = pd.read_csv(sex_excel_fp)
    df = df.rename(columns={"case_id": p_id_column_name})
    df = df.drop(columns=["Unnamed: 0"])
    df.index = df[p_id_column_name]
    df = df.drop(columns=[p_id_column_name])

    return df
# %%
df = parse_sex_info(sex_excel_fp=sex_fp)
display(df.head(n=100))
# %%
sex_series = df["sex"]
value_counts = sex_series.value_counts()

value_counts.plot(kind='bar')
# %%
