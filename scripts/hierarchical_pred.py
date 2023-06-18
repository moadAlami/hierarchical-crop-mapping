import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, RobustScaler
import numpy as np
import pandas as pd
import pickle

df_path = '../data/culture_dataset.parquet'
df = pd.read_parquet(df_path)
df = df.drop(df.query('culture=="olivier"').index)

le_broad = LabelEncoder()
df['filiere'] = le_broad.fit_transform(df['filiere'])
le_crops = LabelEncoder()
df['culture'] = le_crops.fit_transform(df['culture'])

df_train, df_test = df.query('TRAIN==True'), df.query('TRAIN==False')
vis = [f'V{i+1}' for i in range(14)]
bands = [f'B{i+1}' for i in range(140)]

predictor = vis
if predictor == bands:
    shape = (-1, len(bands) // 10, 10)
elif predictor == vis:
    shape = (-1, len(vis), 1)
scaler = RobustScaler()
X_train, X_test = df_train[predictor].values, df_test[predictor].values
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


y_broad_train, y_broad_test = df_train['filiere'], df_test['filiere']
broad_clf = pickle.load(open('../models/groups_XGBClassifier.pickle', 'rb'))
y_broad_pred = broad_clf.predict(X_test)

y_fine_train, y_fine_test = df_train['culture'], df_test['culture']
flat_clf = pickle.load(open('../models/crops_XGBClassifier.pickle', 'rb'))
y_fine_pred_flat = flat_clf.predict(X_test)

fine_clfs = {}
fine_les = {}
for broad_class in set(y_broad_train):
    X_train_subset = X_train[y_broad_train == broad_class]
    X_test_subset = X_test[y_broad_test == broad_class]

    y_fine_train_subset = y_fine_train[y_broad_train == broad_class]
    y_fine_train_subset_decoded = le_crops.inverse_transform(y_fine_train_subset)
    fine_le = LabelEncoder()
    y_fine_train_subset_encoded = fine_le.fit_transform(y_fine_train_subset_decoded)

    y_fine_test_subset = y_fine_test[y_broad_test == broad_class]
    y_fine_test_subset_decoded = le_crops.inverse_transform(y_fine_test_subset)
    y_fine_test_subset_encoded = fine_le.transform(y_fine_test_subset_decoded)

    str_broad_class = le_broad.inverse_transform([broad_class])[0]
    if str_broad_class == 'cereales':
        num_classes = len(set(y_fine_train_subset_encoded))
        y_fine_train_subset_encoded = to_categorical(y_fine_train_subset_encoded)
        y_fine_test_subset_encoded = to_categorical(y_fine_test_subset_encoded)
        X_fine_train_subset = X_train_subset.reshape(shape)
        X_fine_test_subset = X_test_subset.reshape(shape)

        model = load_model('../models/cereales_dl.h5')
        fine_clfs[str_broad_class] = model
    if str_broad_class == 'legumineux':
        fine_clf = pickle.load(open('../models/legumineux_SVC.pickle', 'rb'))
        fine_clfs[str_broad_class] = fine_clf
    elif str_broad_class == 'arboriculture':
        fine_clf = pickle.load(open('../models/arboriculture_RandomForestClassifier.pickle', 'rb'))
        fine_clfs[str_broad_class] = fine_clf

    fine_les[broad_class] = fine_le


y_fine_pred = np.zeros_like(y_fine_test)
for broad_pred in set(y_broad_pred):
    str_broad_class = le_broad.inverse_transform([broad_pred])[0]
    indices = np.where(y_broad_pred == broad_pred)
    if str_broad_class == 'oleagineux':
        y_fine_pred[indices] = le_crops.transform(['colza'])
    elif str_broad_class == 'maraicheres':
        y_fine_pred[indices] = le_crops.transform(['melon'])
    else:
        fine_clf = fine_clfs[str_broad_class]
        fine_le = fine_les[broad_pred]
        if str_broad_class == 'cereales':
            X_test_reshape = X_test.reshape(shape)
            y_initial_pred = fine_clf.predict(X_test_reshape[indices]).argmax(axis=1)
        else:
            y_initial_pred = fine_clf.predict(X_test[indices])
        y_decoded_pred = fine_le.inverse_transform(y_initial_pred)
        y_fine_pred[indices] = le_crops.transform(y_decoded_pred)

print('Broad')
print(classification_report(y_true=y_broad_test, y_pred=y_broad_pred, target_names=le_broad.classes_))
print('Hierarchical')
print(classification_report(y_true=y_fine_test, y_pred=y_fine_pred, target_names=le_crops.classes_))
print('Flat')
print(classification_report(y_true=y_fine_test, y_pred=y_fine_pred_flat, target_names=le_crops.classes_))
