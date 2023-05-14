import geopandas as gpd

reference_poly_path = '../data/gharb-06-2021.shp'
reference_poly = gpd.read_file(reference_poly_path)

gdf_path = '../data/gharb_2021_plots.gpkg'
gdf = gpd.read_file(gdf_path)

oranger_idx = gdf.query("culture.isin(['clementine', 'navel', 'valancia'])").index
gdf.loc[oranger_idx, 'culture'] = 'oranger'


def find_class(key: str) -> str:
    hierarchy = {'cereales': ['ble tendre', 'orge', 'ble dur', 'avoine', 'mais'],
                 'oleagineuses': ['colza', 'tournesol'],
                 'arboriculture': ['grenadier', 'oranger', 'pecher', 'olivier'],
                 'legumineuses': ['pois chiche', 'feverole', 'feve', 'haricot', 'quinoa', 'petit pois', 'lentille'],
                 'jachere': ['jachere'],
                 'maraicheres': ['betterave', 'melon', 'pasteque', 'tomate', 'oignon', 'concombre', 'artichaut'],
                 'fourrageres': ['luzerne', 'bersim'],
                 'aromatiques': ['fenugrec']}
    for class_key, class_value in hierarchy.items():
        if key in class_value:
            return class_key
    return 'Not found'


gdf['filiere'] = None
for culture in gdf.culture.unique():
    gdf.loc[gdf.query('culture == @culture').index, 'filiere'] = find_class(culture)

gdf.crs = reference_poly.crs

gdf.to_file('../data/gharb_2021_plots.shp', encoding='utf8')
