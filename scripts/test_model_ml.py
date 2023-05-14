import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
# import keras as k
from sklearn.metrics import classification_report, f1_score, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

target_class: str = 'culture'
df = pd.read_parquet('../data/ee_sampled_pts_df_2021.parquet')
print(df.culture.value_counts())
# df = df.query('filiere=="cereales"')

def preprocess_data(df: pd.DataFrame, target_class: str = 'culture'):
    columns_to_drop = ['filiere', 'culture', 'TRAIN']
    df_train, df_test = df.query('TRAIN==True'), df.query('TRAIN==False')
    X_train = df_train.drop(columns=columns_to_drop).values
    y_train = df_train[target_class].values
    X_test = df_test.drop(columns=columns_to_drop).values
    y_test = df_test[target_class].values
    # scale
    # X_train, X_test = X_train / 10_000, X_test / 10_000
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # encode
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    classes = label_encoder.classes_
    return X_train, y_train, X_test, y_test, classes


X_train, y_train, X_test, y_test, classes = preprocess_data(df, target_class)

# X_train, X_test = X_train.reshape(-1, 14, 10), X_test.reshape(-1, 14, 10)
# y_train = k.utils.to_categorical(y_train)
# y_test = k.utils.to_categorical(y_test)


# def build_model(x, y):
#     Inputs = k.layers.Input((14, 10))
#     CNN = k.layers.Conv1D(filters=16, kernel_size=1, activation='relu')(Inputs)
#     CNN = k.layers.Flatten()(CNN)
#     CNN = k.layers.Dense(8, activation='relu')(CNN)
#     Outputs = k.layers.Dense(4, activation='softmax')(CNN)
#     model = k.Model(inputs=Inputs, outputs=Outputs)
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['categorical_accuracy'])
#     return model


# model = build_model(X_train, y_train)
# history = model.fit(x=X_train, y=y_train,
#                     epochs=10,
#                     batch_size=256,
#                     shuffle=True,
#                     validation_data=(X_test, y_test),
#                     verbose=0)
# y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
# y_test = y_test.argmax(axis=1)

model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_true=y_test,
                            y_pred=y_pred,
                            target_names=classes))


def plot_cm(clf, ax, x, y, labels, normalize=None):
    y_pred = clf.predict(x)
    if len(labels) == 2:
        avg = 'binary'
    else:
        avg = 'weighted'
    f1 = round(f1_score(y_true=y, y_pred=y_pred, average=avg), 2)
    cm = confusion_matrix(y_true=y, y_pred=y_pred, normalize=normalize)
    sns.heatmap(cm, annot=True, cmap='Blues', ax=ax, fmt='.2f', square='all',
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(f'{clf.__class__.__name__}\nf1-score: {f1}')


fig, ax = plt.subplots(1, figsize=(16, 6))
plot_cm(model, ax, X_test, y_test, classes, normalize='true')
plt.show()

# print(confusion_matrix(y_true=y_test, y_pred=y_pred))


# scores = ['loss', 'categorical_accuracy']
# fig, axs = plt.subplots(1, len(scores))
# c = 0
# for score in scores:
#     axs[c].plot(history.history[f'{score}'], label='Training')
#     axs[c].plot(history.history[f'val_{score}'], label='Validation')
#     axs[c].legend(loc='best')
#     axs[c].set_ylabel(f'{score}'.replace('_', ' ').capitalize())
#     c += 1

# plt.tight_layout()
# plt.show()
