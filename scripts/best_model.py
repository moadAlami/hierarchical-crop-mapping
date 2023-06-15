from ml_utils import pipeline_gridsearch
import pandas as pd


df = pd.read_parquet('../data/culture_dataset_v2.parquet')
df = df.dropna()
to_drop = ['olivier', 'grenadier']
df = df.drop(df.query('culture.isin(@to_drop)').index)

groups = []
for group in df.filiere.unique():
    crops = df.query('filiere==@group').culture.unique()
    if len(crops) > 1:
        groups.append(group)

pipeline_gridsearch(df=df, target_class='filiere')

for group in groups:
    group_df = df.query(f'filiere=="{group}"')
    pipeline_gridsearch(df=group_df, target_class='culture')

pipeline_gridsearch(df=df, target_class='culture')
