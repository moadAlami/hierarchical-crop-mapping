from ml_utils import pipeline_gridsearch
import pandas as pd

df = pd.read_parquet('../data/culture_dataset.parquet')
df = df.drop(df.query('culture=="olivier"').index)
df = df.dropna()

groups = []
for group in df.filiere.unique():
    crops = df.query('filiere==@group').culture.unique()
    if len(crops) > 1:
        groups.append(group)
vis = [f'V{i+1}' for i in range(14)]

pipeline_gridsearch(features=vis,
                    df=df,
                    target_class='filiere',)

for group in groups:
    pipeline_gridsearch(features=vis,
                        df=df,
                        target_class='culture',
                        group_name=group)

pipeline_gridsearch(features=vis,
                    df=df,
                    target_class='culture',)
