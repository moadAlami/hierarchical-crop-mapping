import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import time
import os

df_path = '../data/gharb_pts_2021.parquet'
df = pd.read_parquet(df_path)

# GridSearch params
param_rf = {'n_estimators': [10, 25, 50, 100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [5, 10, 15, 20, 25, None],
            'min_samples_leaf': [1, 2, 5, 10]}
param_svm = {'C': [0.1, 1, 10, 100],
             'gamma': [1, 0.1, 0.01, 0.001],
             'kernel': ['rbf', 'poly', 'sigmoid']}


def preprocess_data(df: pd.DataFrame, target_class: str = 'CROP'):
    columns_to_drop = ['CROP', 'GROUP', 'TRAIN']
    df_train, df_test = df.query('TRAIN==True'), df.query('TRAIN==False')
    X_train = df_train.drop(columns=columns_to_drop).values
    y_train = df_train[target_class].values
    X_test = df_test.drop(columns=columns_to_drop).values
    y_test = df_test[target_class].values
    # scale
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # encode
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    classes = label_encoder.classes_
    return X_train, y_train, X_test, y_test, classes


def pipeline(df: pd.DataFrame, target_class: str = 'CROP'):
    if not os.path.exists('../models'):
        os.mkdir('../models')
    if not os.path.exists('../fig'):
        os.mkdir('../fig')
    start = time.time()
    if target_class == 'GROUP':
        group = 'groups'
    elif df.GROUP.unique().shape[0] == 1:
        group = df.GROUP.unique()[0]
    else:
        group = 'crops'
    print(f'Processing {group}..')
    X_train, y_train, X_test, y_test, classes = preprocess_data(df, target_class)
    # best classifiers
    classifiers = [SVC(), RandomForestClassifier()]
    param_grids = [param_svm, param_rf]
    fig, axs = plt.subplots(1, len(classifiers), figsize=(16, 6))
    for classifier, param_grid in zip(classifiers, param_grids):
        clf = get_best_model(classifier, param_grid, X_train, y_train)
        model_name = f'../models/{group}_{clf.__class__.__name__}.pickle'
        pickle.dump(clf, open(model_name, 'wb'))
        ax = axs[classifiers.index(classifier)]
        plot_cm(clf, ax, X_test, y_test, classes, normalize='true')
        plt.savefig(f'../fig/{group}_cm.png', dpi=150)
    end = time.time()
    exec_time = int(round(end - start, 0))
    print(f'\tDone! ({exec_time} seconds)')


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


def get_best_model(clf, param_grid, x, y):
    model = GridSearchCV(estimator=clf,
                         param_grid=param_grid,
                         refit=True,
                         cv=3,
                         n_jobs=-1,
                         error_score='raise',
                         verbose=0)
    model.fit(x, y)
    best_model = model.best_estimator_
    return best_model


df = df.query('GROUP=="Cereals"')
pipeline(df, target_class='CROP')

pipeline(df=df, target_class='GROUP')

groups = ['Leguminous', 'Fruits', 'Cereals']
for group in groups:
    group_df = df.query(f'GROUP=="{group}"')
    pipeline(df=group_df, target_class='CROP')

pipeline(df=df, target_class='CROP')
