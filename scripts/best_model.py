from ml_utils import pipeline_gridsearch
import pandas as pd

df_path = '../data/ee_sampled_pts_df_2021.parquet'
df = pd.read_parquet(df_path)
df = df.query('filiere=="arboriculture"')

# pipeline_gridsearch(df=df, target_class='filiere')

# groups = ['legumineuses', 'arboriculture', 'cereales', 'maraicheres']
# for group in groups:
#     group_df = df.query(f'filiere=="{group}"')
#     pipeline_gridsearch(df=group_df, target_class='culture')

pipeline_gridsearch(df=df, target_class='culture')
