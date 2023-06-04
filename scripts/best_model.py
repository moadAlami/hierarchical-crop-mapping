from ml_utils import pipeline_gridsearch
import pandas as pd


df_filiere_path = '../data/filiere_dataset.parquet'
df_filiere = pd.read_parquet(df_filiere_path)
df_filiere['culture'] = None

pipeline_gridsearch(df=df_filiere, target_class='filiere')

df_culture_path = '../data/culture_dataset.parquet'
df_culture = pd.read_parquet(df_culture_path)
df_culture = df_culture.drop(df_culture.query('culture=="avoine"').index)


groups = ['cereales', 'fruitiers', 'legumineuses']
for group in groups:
    group_df = df_culture.query(f'filiere=="{group}"')
    pipeline_gridsearch(df=group_df, target_class='culture')

pipeline_gridsearch(df=df_culture, target_class='culture')
