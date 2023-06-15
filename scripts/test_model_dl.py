import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from ml_utils import get_xy
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from keras.layers import Input, Conv1D, Dense, Flatten, Dropout, concatenate, BatchNormalization
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf
import random

tf.random.set_seed(99)
random.seed(99)

df = pd.read_parquet('../data/culture_dataset_v2.parquet')
df = df.query('filiere=="cereales"')

X_train, X_test, y_train, y_test, le = get_xy(df, 'culture')

ndvi_train = X_train[:, :14].reshape(-1, 14, 1)
ndvi_test = X_test[:, :14].reshape(-1, 14, 1)

X_train = X_train[:, 14:].reshape(-1, 14, 10)
X_test = X_test[:, 14:].reshape(-1, 14, 10)

y_train, y_test = to_categorical(y_train), to_categorical(y_test)
target_names = le.classes_
dropout = .2
kernel_size = 1
ndvi_inputs = Input((14, 1))
ndvi_branch = Conv1D(filters=16, kernel_size=kernel_size, activation='relu')(ndvi_inputs)
ndvi_branch = Dropout(dropout)(ndvi_branch)
ndvi_branch = Conv1D(filters=8, kernel_size=kernel_size, activation='relu')(ndvi_branch)

xs_inputs = Input((14, 10))
xs_branch = Conv1D(filters=16, kernel_size=kernel_size, activation='relu')(xs_inputs)
xs_branch = Dropout(dropout)(xs_branch)
xs_branch = Conv1D(filters=8, kernel_size=kernel_size, activation='relu')(xs_branch)
xs_branch = BatchNormalization()(xs_branch)

X = concatenate([ndvi_branch, xs_branch], axis=1)
X = Flatten()(X)
X = Dense(units=16, activation='relu')(X)
X = Dropout(.3)(X)
X = Dense(units=8, activation='relu')(X)
outputs = Dense(units=len(target_names), activation='softmax')(X)
model = Model(inputs=[ndvi_inputs, xs_inputs], outputs=outputs)

metrics = ['Recall', 'Precision']
model.compile(optimizer=Adam(learning_rate=.005, decay=.2),
              metrics=metrics,
              loss='categorical_crossentropy')

es = EarlyStopping(monitor='val_loss',
                   patience=10,
                   mode='min',
                   restore_best_weights=True,
                   verbose=1)

history = model.fit(x=[ndvi_train, X_train], y=y_train,
                    batch_size=512,
                    epochs=100,
                    shuffle=True,
                    callbacks=[es],
                    validation_data=([ndvi_test, X_test], y_test),
                    verbose=0)

y_true = y_test.argmax(axis=1)
y_pred = model.predict([ndvi_test, X_test], verbose=0).argmax(axis=1)

print(classification_report(y_true=y_true, y_pred=y_pred, target_names=target_names))
print(confusion_matrix(y_true=y_true, y_pred=y_pred))

scores = ['recall', 'precision', 'loss']
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
