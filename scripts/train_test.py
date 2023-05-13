import geopandas as gpd

gdf = gpd.read_file('../data/gharb_2021_plots.shp', encoding='utf8')

print(gdf.query('culture=="avoine"'))
