import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


target_class: str = 'filiere'
df = pd.read_parquet('../data/ee_sampled_pts_df_2021.parquet')


def preprocess_data(df: pd.DataFrame, target_class: str = 'culture'):
    if target_class == 'culture':
        train = 'TRAIN'
    elif target_class == 'filiere':
        train = 'G_TRAIN'
    columns_to_drop = ['filiere', 'culture', 'TRAIN', 'G_TRAIN']
    df_train, df_test = df.query(f'{train}==True'), df.query(f'{train}==False')
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

model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_true=y_test, y_pred=y_pred, target_names=classes))
print(confusion_matrix(y_true=y_test, y_pred=y_pred))
