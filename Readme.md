# Hierarchical crop mapping

## Data
  - gharb_2021_plots.shp: The original survey polygon (gharb-06-2021) which was split, was used as a reference to manually digitize agricultural plots. The 'culture' column was joined to the final polygon file.
    Train test was first done manually in qgis by spliting the dataset after looking at the number of pixel per polygon. 
    The number of pixels in a polygon is roughly area (ha) / 100
  - gharb_2021_pts.gpkg: points originally sampled and inside gharb parcels (not the one mentionned above) and has 140 columns (10 Sentinel-2 bands, 14 dates).
    Additionally, it has the columns: CROP, GROUP and TRAIN.
    The TRAIN column is taken from 'gharb_2021_plots.shp'.
    The geometry column was dropped and band columns were converted to int16 to create 'gharb_2021_pts.parquet'
  
  ### Important
  - Uploading shp to ee with ESRI:102191 results in a noticeable shift.
  - gpkg handles projections in a weird mannger. Use shp and then parquet.
    
## TODO
 - [X] Pre-process shapefiles and only keep parquet in the repo 
 - [ ] Correct typo pomegranate
 - [ ] Add pairplot with NDVI for cereal (`sns.pairplot`)
 - [ ] Remove models/ from .gitignore


