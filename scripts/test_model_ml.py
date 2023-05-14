import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

target_class: str = 'CROP'
df = pd.read_file('../data/gharb_2021_pts.parquet')


def preprocess_data(df: pd.DataFrame, target_class: str = 'CROP'):
    columns_to_drop = ['GROUP', 'CROP', 'TRAIN']
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

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_true=y_test,
                            y_pred=y_pred,
                            target_names=classes))
