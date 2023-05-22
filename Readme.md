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

## Notes
- It can be more efficient to train and fine tune deep learning models than classical machine learning models because deep learning models run on GPUs with large batch size.
- Misclassifications in group classification have major repercussions on fine crop classification. This is more so obvious when there are omission errors in an under represented broad class.
- In earthengine, filtering the collection with < 1% cloud cover lowers the accuracy. Dates had to be identiafied manually.
- SVM classification with kernel 'sigmoid' takes too long in comparison with 'rbf' and 'poly' and it has the lowest accuracy.
- XGBoost takes the longest, SVM is close behind, and RF takes much lower (even with more fits than SVM).
- It should be noted that the problem of a hierarchy that does not respect the parent child relationship is absent. Unlike a study where multi-label classes were assigned, our scheme uses a conditional approach that guarantees the cohesiveness of thematic classes.
- The initial train test split is crucial for a successful pipeline run. It is important to split the polygons rather than the pixels to avoid spatial autocorrelation. 
    
## TODO
 - [X] Move common functions to ml_utils
 - [X] Make custom GridSearch with test dataset instead of cross validation
 - [X] Train fine crop classifiers
 - [ ] For hierarchical_pred, perform a broad class classification, then get the index of each class and apply the appropriate fine classifier. Compare execution time with iterating over each pixel.

### Later
 - [ ] Automatic train test split that ensures a good ratio of train/test while avoiding spatial autocorrelation.
 - [ ] Automatic identification of the optimal dates
