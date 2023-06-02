from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv1D, LSTM, Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from ml_utils import get_xy
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(9)

target_class = 'culture'
df = pd.read_parquet('../data/culture_dataset.parquet')
df = df.query('filiere=="cereales"').copy()

NDVI = df.copy()
for i in range(14):
    if i == 0:
        NDVI.loc[NDVI.index, f'V{i+1}'] = (df['B7'] - df['B3']) / (df['B7'] + df['B3'])
    else:
        NDVI.loc[NDVI.index, f'V{i+1}'] = (df[f'B{i}7'] - df[f'B{i}3']) / (df[f'B{i}7'] + df[f'B{i}3'])

ndvi_names = [f'V{i+1}' for i in range(14)]
NDVI = NDVI[ndvi_names]
NDVI = NDVI.dropna()
# bands = [f'B{i+1}' for i in range(140)]
# df = df.loc[NDVI.index]

# merged_df = pd.concat([NDVI, df], axis=1)

X_train, X_test, y_train, y_test, label_encoder = get_xy(df, target_class)
classes = label_encoder.classes_
num_classes = len(classes)
print(classes)

X_shape = (-1, 14, 10)
X_train = X_train.reshape(X_shape)
X_test = X_test.reshape(X_shape)

y_train, y_test = to_categorical(y_train), to_categorical(y_test)

inputs = Input((14, 10))
hidden = Conv1D(filters=128, kernel_size=1, activation='relu')(inputs)
hidden = Conv1D(filters=256, kernel_size=1, activation='relu')(hidden)
hidden = Dropout(.4)(hidden)
hidden = Flatten()(hidden)
outputs = Dense(units=num_classes, activation='softmax')(hidden)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=.005, decay=.2),
              metrics='categorical_accuracy',
              loss='categorical_crossentropy')

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
