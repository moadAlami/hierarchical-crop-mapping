import geopandas as gpd
import pandas as pd

gdf_path = '../data/GHARB_PARCELS.shp'
gdf = gpd.read_file(gdf_path, encoding='utf8')
gdf = gdf.drop('T_Gr', axis=1)

to_keep = ['Barley', 'Broad beans', 'Common wheat', 'Durum wheat', 'Oats', 'Rapeseed']
# print(pd.concat([gdf.groupby('CROP').COUNT.sum(), gdf.CROP.value_counts()], axis=1))
gdf = gdf.query('CROP.isin(@to_keep)')


def split(geodataframe):
    d = {}
    for crop in geodataframe.CROP.unique():
        cond = False
        while not cond:
            d[crop] = geodataframe.query('CROP==@crop')
            subset_1 = d[crop].sample(frac=.75)
            subset_2 = d[crop].drop(subset_1.index)
            if subset_1.COUNT.sum() > subset_2.COUNT.sum():
                subset_1['TRAIN'] = True
                subset_2['TRAIN'] = False
            else:
                subset_1['TRAIN'] = False
                subset_2['TRAIN'] = True
            d[crop] = pd.concat([subset_1, subset_2])
            lim_min = 20 * d[crop].COUNT.sum() / 100
            lim_max = 30 * d[crop].COUNT.sum() / 100
            test_pixels = d[crop].query('TRAIN==False').COUNT.sum()
            if lim_min <= test_pixels <= lim_max:
                cond = True
    full_df = pd.concat([d[i] for i in d])
    return full_df


new_gdf = split(gdf).reset_index(drop=True)
new_gdf.to_file('../data/GHARB_PARCELS_FREQ.shp', encoding='utf8')
