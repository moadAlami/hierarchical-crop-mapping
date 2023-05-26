import geopandas as gpd
import pandas as pd
import random

gdf_path: str = r'../../data/gharb_2021_plots_wgs.shp'
gdf: gpd.GeoDataFrame = gpd.read_file(gdf_path, encoding='utf8')
gdf = gdf.rename(mapper={'area': 'AREA'}, axis=1)
gdf = gdf.drop(columns=['G_TRAIN', 'TRAIN'])


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
        upper_lim = 0.8
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


cultures = ['ble tendre', 'orge', 'ble dur', 'avoine', 'colza', 'grenadier', 'oranger', 'pois chiche', 'feverole', 'melon', 'oignon']

d: dict = {}
for culture in cultures:
    d[culture] = split(gdf.query('culture==@culture'))

full_df = pd.concat([d[i] for i in d])

full_df = full_df.drop('AREA', axis=1)
out_gdf = r'../../data/gharb_2021_plots_wgs_v2.shp'

full_df.to_file(out_gdf, encoding='utf8')
