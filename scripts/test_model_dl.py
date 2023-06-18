import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv1D, Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler, LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import random
import tensorflow as tf

seed = 99
tf.random.set_seed(seed)
random.seed(seed)

df = pd.read_parquet('../data/culture_dataset.parquet')
le = LabelEncoder()
le.fit(df.query('filiere=="cereales"')['culture'])
num_classes = len(le.classes_)
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
scaler.fit(X_train)
X_train = df_train.query('filiere=="cereales"')[predictor].values
X_train = scaler.transform(X_train).reshape(shape)
X_test = df_test.query('filiere=="cereales"')[predictor].values
X_test = scaler.transform(X_test).reshape(shape)

y_train = le.transform(df_train.query('filiere=="cereales"')['culture'])
y_train = to_categorical(y_train)
y_test = le.transform(df_test.query('filiere=="cereales"')['culture'])
y_test = to_categorical(y_test)

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
optimizer = Adam(decay=0.01, learning_rate=0.001)

inputs = Input(shape[1:])

hidden = Conv1D(filters=16, kernel_size=3, activation='relu')(inputs)
hidden = Conv1D(filters=16, kernel_size=3, activation='relu')(hidden)
hidden = Conv1D(filters=8, kernel_size=2, activation='relu')(hidden)
hidden = Flatten()(hidden)

outputs = Dense(units=num_classes, activation='softmax')(hidden)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])
history = model.fit(x=X_train, y=y_train,
                    batch_size=256,
                    epochs=200,
                    callbacks=es,
                    shuffle=True,
                    validation_data=(X_test, y_test),
                    verbose=1)
model.save('../models/cereales_dl.h5')

y_test = y_test.argmax(axis=1)
y_pred = model.predict(X_test, verbose=1).argmax(axis=1)

print(classification_report(y_true=y_test, y_pred=y_pred, target_names=le.classes_))
print(confusion_matrix(y_true=y_test, y_pred=y_pred))

scores = ['recall', 'precision', 'accuracy', 'loss']
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
