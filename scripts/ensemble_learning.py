from ml_utils import get_xy, plot_cm
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
import pickle

target_class: str = 'culture'
df = pd.read_parquet('../data/ee_sampled_pts_df_2021.parquet')
to_classify = 'crops'
# df = df.query('filiere==@to_classify')

X_train, X_test, y_train, y_test, label_encoder = get_xy(df, target_class)
classes = label_encoder.classes_
svm = pickle.load(open(f'../models/{to_classify}_SVC.pickle', 'rb'))
rf = pickle.load(open(f'../models/{to_classify}_RandomForestClassifier.pickle', 'rb'))
xgb = pickle.load(open(f'../models/{to_classify}_XGBClassifier.pickle', 'rb'))
eclf = VotingClassifier(estimators=[('svm', svm),
                                    ('rf', rf),
                                    ('xgb', xgb)],
                        voting='hard')


fig, axs = plt.subplots(1, 4)
c = 0
for clf, label in zip([svm, rf, xgb, eclf], ['SVM', 'Random Forest', 'XGBoost', 'Ensemble']):
    print(label)
    if label == 'Ensemble':
        clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    plot_cm(clf, axs[c], X_test, y_test, classes, normalize='true')
    print(classification_report(y_true=y_test,
                                y_pred=y_pred,
                                target_names=classes))
    print(confusion_matrix(y_true=y_test, y_pred=y_pred))
    print('\n')
    c += 1
