import pandas as pd
from pyogrio import read_dataframe
import pickle

poly = read_dataframe('/home/mouad/SSD/College/PhD/missions/2021/06-02-21/shp/preprocessed/parcelles.shp')
to_drop = ['tomate', 'betterave', 'oignon']
poly = poly.drop(poly.query('culture.isin(@to_drop)').index)
pts = read_dataframe('/home/mouad/SSD/College/PhD/missions/2021/06-02-21/shp/preprocessed/pixels.shp')


def split(geodataframe, target):
    d = {}
    for t in geodataframe[target].unique():
        d[t] = geodataframe[geodataframe[target] == t]
        print(t, d[t].COUNT.sum())
        cond = False
        attempts = 0
        while not cond:
            subset_1 = d[t].sample(frac=.75, random_state=3)
            subset_2 = d[t].drop(subset_1.index)
            if subset_1.COUNT.sum() > subset_2.COUNT.sum():
                subset_1['TRAIN'] = True
                subset_2['TRAIN'] = False
            else:
                subset_1['TRAIN'] = False
                subset_2['TRAIN'] = True
            d[t] = pd.concat([subset_1, subset_2])
            lim_min = 15 * d[t].COUNT.sum() / 100
            lim_max = 45 * d[t].COUNT.sum() / 100
            test_pixels = d[t].query('TRAIN==False').COUNT.sum()
            attempts += 1
            if attempts > 100:
                lim_min = 150
            if lim_min <= test_pixels <= lim_max:
                cond = True
    full_df = pd.concat([d[i] for i in d])
    print(full_df.query('TRAIN==False').groupby('culture').COUNT.sum())
    return full_df


poly = split(poly, 'culture').drop('COUNT', axis=1)
['colza' 'avoine' 'agrumes' 'ble tendre' 'feverole' 'melon' 'grenadier'
 'pois chiche' 'ble dur' 'orge' 'olivier']


def find_class(key: str) -> str:
    hierarchy = {'oleagineux': ['colza'],
                 'cereales': ['avoine', 'ble tendre', 'ble dur', 'orge'],
                 'arboriculture': ['agrumes', 'grenadier', 'olivier'],
                 'legumineux': ['feverole', 'pois chiche'],
                 'maraicheres': ['melon']}
    for class_key, class_value in hierarchy.items():
        if key in class_value:
            return class_key
    return 'Not found'


poly['filiere'] = 'xd'
for culture in poly.culture.unique():
    poly.loc[poly.query('culture == @culture').index, 'filiere'] = find_class(culture)

print(poly.filiere.unique())
pickle.dump(poly, open('../data/parcelles_v2.pickle', 'wb'))
