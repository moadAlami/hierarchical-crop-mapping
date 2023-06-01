import pandas as pd
from ml_utils import get_xy
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

target_class: str = 'culture'
df = pd.read_parquet(f'../data/{target_class}_dataset.parquet')

X_train, X_test, y_train, y_test, label_encoder = get_xy(df, target_class)
classes = label_encoder.classes_

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_true=y_test, y_pred=y_pred, target_names=classes))
print(confusion_matrix(y_true=y_test, y_pred=y_pred))
