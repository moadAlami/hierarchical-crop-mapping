# from ml_utils import plot_feature_importances
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

target_class: str = 'filiere'
df = pd.read_parquet('../data/ee_sampled_pts_df_2021.parquet')


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
clf1 = SVC()
clf2 = RandomForestClassifier(50, random_state=1)
clf3 = XGBClassifier()
eclf = VotingClassifier(estimators=[('svm', clf1),
                                    ('rf', clf2),
                                    ('xgb', clf3)],
                        voting='hard')
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)


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


fig, axs = plt.subplots(1, 4)
c = 0
for clf, label in zip([clf1, clf2, clf3, eclf], ['SVM', 'Random Forest', 'XGBoost', 'Ensemble']):
    print(label)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # plot_cm(clf, axs[c], X_test, y_test, classes, normalize='true')
    print(classification_report(y_true=y_test,
                                y_pred=y_pred,
                                target_names=classes))
    print(confusion_matrix(y_true=y_test, y_pred=y_pred))
    print('\n')
    c += 1
# feature_name = df.drop(columns=['TRAIN', 'culture', 'filiere']).columns
# plot_feature_importances(clf=model, feature_names=feature_name, ax=ax)
# plt.tight_layout()
# plt.show()
