import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from ml_utils import get_xy
import geopandas as gpd
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from pyogrio import read_dataframe
from keras.layers import Input, Conv1D, Dense, Flatten, Dropout
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

poly = read_dataframe('/home/mouad/SSD/College/PhD/missions/2021/06-02-21/shp/preprocessed/parcelles.shp')
# to_drop = ['agrumes', 'betterave', 'oignon']
# poly = poly.drop(poly.query('culture.isin(@to_drop)').index)
to_keep = ['avoine', 'ble dur', 'ble tendre', 'orge']
poly = poly.query('culture.isin(@to_keep)')
pts = read_dataframe('/home/mouad/SSD/College/PhD/missions/2021/06-02-21/shp/preprocessed/pixels.shp')


def split(geodataframe, target):
    d = {}
    for t in geodataframe[target].unique():
        d[t] = geodataframe[geodataframe[target] == t]
        print(t, d[t].COUNT.sum())
        cond = False
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
            lim_min = 20 * d[t].COUNT.sum() / 100
            lim_max = 45 * d[t].COUNT.sum() / 100
            test_pixels = d[t].query('TRAIN==False').COUNT.sum()
            if lim_min < 1:
                lim_min = 1
            if lim_min <= test_pixels <= lim_max:
                cond = True
    full_df = pd.concat([d[i] for i in d])
    print(full_df.query('TRAIN==False').groupby('culture').COUNT.sum())
    return full_df


poly = split(poly, 'culture').drop('COUNT', axis=1)

gdf = gpd.sjoin(pts, poly, how='left').drop('index_right', axis=1)
gdf['filiere'] = None
df = gdf.drop('geometry', axis=1)

X_train, X_test, y_train, y_test, le = get_xy(df, 'culture')
target_names = le.classes_
X_train, X_test = X_train.reshape(-1, 14, 1), X_test.reshape(-1, 14, 1)
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

inputs = Input((14, 1))
hidden = Conv1D(filters=16, kernel_size=3, activation='relu')(inputs)
hidden = Dropout(.4)(hidden)
hidden = Conv1D(filters=8, kernel_size=3, activation='relu')(hidden)
hidden = Dropout(.4)(hidden)
hidden = Flatten()(hidden)
outputs = Dense(units=len(target_names), activation='softmax')(hidden)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=Adam(learning_rate=.005, decay=.2),
              metrics='accuracy',
              loss='categorical_crossentropy')

es = EarlyStopping(monitor='val_accuracy',
                   patience=10,
                   mode='max',
                   restore_best_weights=True,
                   verbose=1)

history = model.fit(x=X_train, y=y_train,
                    batch_size=512,
                    epochs=100,
                    callbacks=[es],
                    validation_data=(X_test, y_test),
                    verbose=0)

y_true = y_test.argmax(axis=1)
y_pred = model.predict(X_test, verbose=0).argmax(axis=1)

print(classification_report(y_true=y_true, y_pred=y_pred, target_names=target_names))
print(confusion_matrix(y_true=y_true, y_pred=y_pred))

scores = ['accuracy', 'loss']
fig, axs = plt.subplots(1, len(scores), figsize=(12, 5))
c = 0
for score in scores:
    axs[c].plot(history.history[f'{score}'], label='Training')
    axs[c].plot(history.history[f'val_{score}'], label='Validation')
    axs[c].legend(loc='best')
    axs[c].set_ylabel(f'{score}'.replace('_', ' ').capitalize())
    axs[c].set(title=f'{score}'.replace('_', ' ').capitalize())
    c += 1

plt.show()
