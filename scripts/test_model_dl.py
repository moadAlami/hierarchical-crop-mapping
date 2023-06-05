import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv1D, Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from ml_utils import get_xy
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

tf.random.set_seed(9)

target_class = 'culture'
df = pd.read_parquet('../data/culture_dataset.parquet')
df = df.query('filiere=="cereales"').copy()
df = df.drop(df.query('culture=="avoine"').index)

X_train, X_test, y_train, y_test, label_encoder = get_xy(df, target_class)
classes = label_encoder.classes_
num_classes = len(classes)
print(classes)

X_shape = (-1, 14, 10)
X_train = X_train.reshape(X_shape)
X_test = X_test.reshape(X_shape)

y_train, y_test = to_categorical(y_train), to_categorical(y_test)

metrics = ['accuracy', 'Recall', 'Precision']

inputs = Input((14, 10))
hidden = Conv1D(filters=128, kernel_size=1, activation='relu')(inputs)
hidden = Conv1D(filters=256, kernel_size=1, activation='relu')(hidden)
hidden = Dropout(.4)(hidden)
hidden = Flatten()(hidden)
outputs = Dense(units=num_classes, activation='softmax')(hidden)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=.005, decay=.2),
              metrics=metrics,
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
                    verbose=0)

# model.save('../models/cereal_dl.h5')

y_true = y_test.argmax(axis=1)
y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
report = classification_report(y_true=y_true, y_pred=y_pred, target_names=classes)
print(report)
cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
print(cm)

metrics = [i.lower() for i in metrics]
scores = ['loss'] + metrics
fig, axs = plt.subplots(1, len(scores), figsize=(12, 5))
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
