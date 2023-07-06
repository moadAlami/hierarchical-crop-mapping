import numpy as np
import rasterio as rio
import pickle
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model

df = pd.read_parquet('../data/culture_dataset.parquet')

vis = [f'V{i+1}' for i in range(14)]

sc = pickle.load(open('../assets/scalers.pickle', 'rb'))
le = pickle.load(open('../assets/label_encoders.pickle', 'rb'))
clf = {}
clf['groups'] = pickle.load(open('../models/groups_XGBClassifier.pickle', 'rb'))
clf['legumineux'] = pickle.load(open('../models/legumineux_SVC.pickle', 'rb'))
clf['arboriculture'] = pickle.load(open('../models/arboriculture_RandomForestClassifier.pickle', 'rb'))
clf['cereales'] = load_model('../models/cereales_dl.h5')

img_path = '../data/ndvi-subset.tif'
with rio.open(img_path) as src:
    img = src.read()
    meta = src.meta

bands, height, width = img.shape
ndvi = img.transpose(1, 2, 0).reshape(-1, bands)
ndvi_scaled = sc['groups'].transform(ndvi)

y_broad_pred = clf['groups'].predict(ndvi_scaled)

y_fine_pred = np.zeros_like(y_broad_pred)
y_confidence = np.zeros_like(y_broad_pred, dtype=float)
for broad_pred in set(y_broad_pred):
    le_broad = le['groups']
    le_crops = le['crops']
    str_broad_class = le_broad.inverse_transform([broad_pred])[0]
    indices = np.where(y_broad_pred == broad_pred)
    if str_broad_class == 'oleagineux':
        y_fine_pred[indices] = le_crops.transform(['colza'])
        y_confidence[indices] = 1
    elif str_broad_class == 'maraicheres':
        y_fine_pred[indices] = le_crops.transform(['melon'])
        y_confidence[indices] = 1
    else:
        fine_clf = clf[str_broad_class]
        fine_le = le[str_broad_class]
        if str_broad_class == 'cereales':
            ndvi_scaled_reshaped = ndvi_scaled.reshape(-1, 14, 1)
            y_initial_pred = fine_clf.predict(ndvi_scaled_reshaped[indices])
            y_confidence[indices] = np.max(y_initial_pred, axis=1)
            y_initial_pred = y_initial_pred.argmax(axis=1)
        else:
            y_initial_pred = fine_clf.predict(ndvi_scaled[indices])
            y_confidence[indices] = np.max(fine_clf.predict_proba(ndvi_scaled[indices]), axis=1)
        y_decoded_pred = fine_le.inverse_transform(y_initial_pred)
        y_fine_pred[indices] = le_crops.transform(y_decoded_pred)

# original shape
final_pred = y_fine_pred.reshape(height, width, 1).transpose(2, 0, 1)[0]
final_pred = final_pred.reshape(1, height, width)

final_confid = y_confidence.reshape(height, width, 1).transpose(2, 0, 1)[0]
final_confid = final_confid.reshape(1, height, width)

final_broad = y_broad_pred.reshape(height, width, 1).transpose(2, 0, 1)[0]
final_broad = final_broad.reshape(1, height, width)

meta.update(count=1, dtype=str(final_broad.dtype))
output_path = '../output/broad.tif'
with rio.open(output_path, 'w', **meta) as dst:
    dst.write(final_broad)

meta.update(count=1, dtype=str(final_pred.dtype))
output_path = '../output/prediction.tif'
with rio.open(output_path, 'w', **meta) as dst:
    dst.write(final_pred)

meta.update(count=1, dtype=str(final_confid.dtype))
output_path = '../output/confidence.tif'
with rio.open(output_path, 'w', **meta) as dst:
    dst.write(final_confid)
