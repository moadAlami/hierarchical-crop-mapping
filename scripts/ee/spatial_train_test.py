import geopandas as gpd
import pandas as pd
import random

gdf_path: str = r'../../data/survey_2021_wgs.shp'
gdf: gpd.GeoDataFrame = gpd.read_file(gdf_path, encoding='utf8')
gdf = gdf.rename(mapper={'area': 'AREA'}, axis=1)


def split(gdf):
    sum_area: float = gdf.AREA.sum()
    if sum_area == 0:
        return None
    cond = False
    while not cond:
        frac: float = round(random.uniform(.6, .95), 2)
        subset_1: gpd.GeoDataFrame = gdf.sample(frac=frac)
        subset_2 = gdf.drop(subset_1.index)
        ratio_1: float = round(subset_1.AREA.sum() / sum_area, 2)
        ratio_2: float = round(subset_2.AREA.sum() / sum_area, 2)
        min_poly = 2
        if gdf.shape[0] == 3:
            min_poly = 1
        lower_lim = 0.7
        upper_lim = 0.9
        if lower_lim < ratio_1 < upper_lim and subset_2.shape[0] >= min_poly:
            subset_1['TRAIN'] = True
            subset_2['TRAIN'] = False
            cond = True
        elif lower_lim < ratio_2 < upper_lim and subset_1.shape[0] >= min_poly:
            subset_1['TRAIN'] = False
            subset_2['TRAIN'] = True
            cond = True
    new_gdf = pd.concat([subset_1, subset_2], axis=0)
    return new_gdf


cultures_by_number = gdf.culture.value_counts()[gdf.culture.value_counts() > 1].index
cultures_by_area = gdf.groupby('culture').AREA.sum()[gdf.groupby('culture').AREA.sum() > 300].index
cultures = set(cultures_by_number).intersection(set(cultures_by_area))
print(cultures)
d: dict = {}
for culture in cultures:
    d[culture] = split(gdf.query('culture==@culture'))

full_df = pd.concat([d[i] for i in d])

full_df = full_df.drop('AREA', axis=1)

for culture in cultures:
    train = round(d[culture].query('TRAIN==True').AREA.sum(), 2)
    test = round(d[culture].query('TRAIN==False').AREA.sum(), 2)
    print(f'{culture}: {train}/{test}')

out_gdf = r'../../data/survey_2021_wgs_split.shp'

full_df.to_file(out_gdf, encoding='utf8')
