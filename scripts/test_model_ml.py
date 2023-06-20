from ml_utils import get_xy
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix

target_class: str = 'culture'
df = pd.read_parquet('../data/culture_dataset.parquet')
df = df.drop(df.query('culture=="olivier"').index)

vis = [f'V{i+1}' for i in range(14)]

X_train, X_test, y_train, y_test, le = get_xy(df=df, features=vis, group_name='cereales')
classes = le.classes_

clfs = ['DecisionTreeClassifier', 'RandomForestClassifier', 'SVC', 'XGBClassifier']

for clf_name in clfs:
    clf = pickle.load(open(f'../models/cereales_{clf_name}.pickle', 'rb'))
    y_pred = clf.predict(X_test)
    print(clf_name)
    print(classification_report(y_true=y_test, y_pred=y_pred, target_names=classes))
