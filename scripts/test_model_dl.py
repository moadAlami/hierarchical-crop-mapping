from keras.layers import Input, Conv1D, Flatten, Dense
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from ml_utils import get_xy
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(9)

df = pd.read_parquet('../data/ee_sampled_pts_df_2021.parquet')
df = df.query('filiere=="cereales"')
non_bands = ['filiere', 'culture', 'TRAIN', 'G_TRAIN']
old_names = df.drop(columns=non_bands).columns.tolist()
new_names = [f'B{i+1}' for i in range(len(old_names))]

mapper = {}
for old, new in zip(old_names, new_names):
    mapper[old] = new

df = df.rename(mapper=mapper, axis=1)

NDVI = df.copy()
for i in range(len(new_names) // 10):
    if i == 0:
        NDVI.loc[NDVI.index, f'V{i+1}'] = (df['B7'] - df['B3']) / (df['B7'] + df['B3'])
    else:
        NDVI.loc[NDVI.index, f'V{i+1}'] = (df[f'B{i}7'] - df[f'B{i}3']) / (df[f'B{i}7'] + df[f'B{i}3'])

ndvi_names = [f'V{i+1}' for i in range(len(new_names) // 10)]
NDVI = NDVI[ndvi_names]
NDVI = NDVI.dropna()
df = df.loc[NDVI.index]

merged_df = pd.concat([NDVI, df], axis=1)

X_train, X_test, y_train, y_test, label_encoder = get_xy(merged_df, 'culture')
classes = label_encoder.classes_
num_classes = len(classes)

X_shape = (-1, X_train.shape[1], 1)
X_train = X_train.reshape(X_shape)
X_test = X_test.reshape(X_shape)

y_train, y_test = to_categorical(y_train), to_categorical(y_test)


def build_model(x_shape: tuple, num_classes: int):
    inputs = Input(x_shape)
    hidden = Conv1D(filters=8, kernel_size=1)(inputs)
    # hidden = Dropout(.2)(hidden)
    hidden = Flatten()(hidden)
    # hidden = Dense(units=32, activation='relu')(hidden)
    outputs = Dense(units=num_classes, activation='softmax')(hidden)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  metrics='categorical_accuracy',
                  loss='categorical_crossentropy')
    return model


model = build_model(X_train.shape[1:], num_classes)

es = EarlyStopping(monitor='val_loss',
                   patience=10,
                   mode='min',
                   restore_best_weights=True,
                   verbose=1)

history = model.fit(x=X_train, y=y_train,
                    batch_size=512,
                    epochs=100,
                    callbacks=[es],
                    validation_data=(X_test, y_test),
                    verbose=1)

y_true = y_test.argmax(axis=1)
y_pred = model.predict(X_test).argmax(axis=1)
report = classification_report(y_true=y_true, y_pred=y_pred, target_names=classes)
print(report)
cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
print(cm)

scores = ['loss', 'categorical_accuracy']
fig, axs = plt.subplots(1, len(scores))
c = 0
for score in scores:
    axs[c].plot(history.history[f'{score}'], label='Training')
    axs[c].plot(history.history[f'val_{score}'], label='Validation')
    axs[c].legend(loc='best')
    axs[c].set_ylabel(f'{score}'.replace('_', ' ').capitalize())
    axs[c].set(title=f'{score}'.replace('_', ' ').capitalize())
    c += 1

plt.tight_layout()
plt.show()
