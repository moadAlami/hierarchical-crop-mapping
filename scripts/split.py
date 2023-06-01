import geopandas as gpd
import pandas as pd

gdf_path = '../data/GHARB_PARCELS.shp'
gdf = gpd.read_file(gdf_path, encoding='utf8')
to_drop = gdf.groupby('culture').COUNT.sum()[gdf.groupby('culture').COUNT.sum() < 300].index
gdf.loc[gdf.query('culture.isin(@to_drop)').index, 'use'] = 1
target = 'culture'
if target == 'culture':
    thresh = 1
elif target == 'filiere':
    thresh = 0
gdf = gdf.query('use > @thresh').copy()


def split(geodataframe, target):
    d = {}
    for t in geodataframe[target].unique():
        d[t] = geodataframe[geodataframe[target] == t]
        print(t, d[t].COUNT.sum())
        cond = False
        while not cond:
            subset_1 = d[t].sample(frac=.75)
            subset_2 = d[t].drop(subset_1.index)
            if subset_1.COUNT.sum() > subset_2.COUNT.sum():
                subset_1['TRAIN'] = True
                subset_2['TRAIN'] = False
            else:
                subset_1['TRAIN'] = False
                subset_2['TRAIN'] = True
            d[t] = pd.concat([subset_1, subset_2])
            lim_min = 20 * d[t].COUNT.sum() / 100
            lim_max = 45 * d[t].COUNT.sum() / 100
            test_pixels = d[t].query('TRAIN==False').COUNT.sum()
            if lim_min <= test_pixels <= lim_max:
                cond = True
    full_df = pd.concat([d[i] for i in d])
    return full_df


new_gdf = split(gdf, target).reset_index(drop=True)

train_count = new_gdf.query('TRAIN==True').groupby(target).COUNT.sum()
test_count = new_gdf.query('TRAIN==False').groupby(target).COUNT.sum()

print(pd.concat([train_count, test_count], axis=1))

new_gdf.to_file('../data/GHARB_PARCELS_FREQ_CU.shp', encoding='utf8')
