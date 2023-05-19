import pandas as pd
from ml_utils import get_xy
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


target_class: str = 'filiere'
df = pd.read_parquet('../data/ee_sampled_pts_df_2021.parquet')


X_train, y_train, X_test, y_test, label_encoder = get_xy(df, target_class)
classes = label_encoder.classes_

model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_true=y_test, y_pred=y_pred, target_names=classes))
print(confusion_matrix(y_true=y_test, y_pred=y_pred))
