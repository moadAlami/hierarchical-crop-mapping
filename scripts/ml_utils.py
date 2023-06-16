import itertools
from functools import partial
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVC
from typing import Tuple, List, Dict
from xgboost import XGBClassifier
import time


def reflectance_plot(df: pd.DataFrame, target_class: str, H: int, W: int) -> None:
    wave = [.490, .560, .665, .705, .740, .783, .842, .865, 1.610, 2.190]
    bands = df.drop(columns=['culture', 'filiere', 'TRAIN']).columns
    dates = []
    for column in df[bands].columns:
        date = f'{column[:4]}-{column[4:6]}-{column[6:8]}'
        if date not in dates:
            dates.append(date)
    labels = df[f'{target_class}'].unique().tolist()
    fig, axs = plt.subplots(H, W, figsize=(W + 6, H + 6))
    c = 0
    idx = 0
    try:
        while c < len(bands):
            for h in range(H):
                for w in range(W):
                    for label in labels:
                        x = df[df[f'{target_class}'] == label][bands[c:c + 10]]
                        x.columns = wave[0:10]
                        x = x.median() / 10_000
                        x.plot(kind='line',
                               linestyle='dashdot',
                               label=label.capitalize(),
                               ax=axs[h, w])
                        axs[h, w].scatter(wave[0:10], x.tolist(), marker='x')
                        axs[h, w].set(title=f'{dates[idx]}',
                                      ylim=(0, .62),
                                      xlabel='Wavelength (µm)',
                                      ylabel='Reflectance')
                    idx += 1
                    c += 10
    except ValueError:
        pass
    Line, Label = axs[0, 0].get_legend_handles_labels()
    fig.legend(Line, Label, loc='lower right',
               bbox_to_anchor=(0.95, 0.055), fontsize=12)
    plt.tight_layout()


def ndvi_plot(ndvi: pd.DataFrame, target_class: str, dates: list):
    """Plot the NDVI values for each label in a given column over time.

    Args:
        ndvi (pd.DataFrame): A pandas DataFrame containing the NDVI values.
        column (str): The name of the column to group by.

    Returns:
        None
    """
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ndvi[f'{target_class}'].unique().tolist()
    for label in labels:
        df = ndvi[ndvi[f'{target_class}'] == label].median(numeric_only=True)
        df.plot(label=label, linestyle='dashdot', ax=ax)
        x = [i for i in range(14)]
        y = df.tolist()
        ax.scatter(x, y, marker='x')
    ax.set(ylabel='NDVI')
    ax.set_xticks([i for i in range(14)])
    ax.set_xticklabels(dates, rotation=90)
    Line, Label = ax.get_legend_handles_labels()
    fig.legend(Line, Label, loc='lower right',
               bbox_to_anchor=(1.05, 0.15),
               fontsize=12)
    # plt.tight_layout()


def get_xy(df: pd.DataFrame, target_class: str = 'culture') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """
     Preprocesses a given pandas DataFrame and returns the features and labels of the training and test sets.

     Args:
         df (pandas.DataFrame): The input DataFrame.
         target_class (str): The name of the target class. Defaults to 'CROP'.

     Returns:
         A tuple containing:
         - X_train (numpy.ndarray): The features of the training set.
         - X_test (numpy.ndarray): The features of the test set.
         - y_train (numpy.ndarray): The labels of the training set.
         - y_test (numpy.ndarray): The labels of the test set.
         - label_encoder (sklearn.preprocessing.LabelEncoder): The label encoder object used to encode the classes.
     """
    columns_to_drop = ['culture', 'filiere', 'TRAIN']
    df_train, df_test = df.query('TRAIN==True'), df.query('TRAIN==False')
    X_train, y_train = df_train.drop(columns=columns_to_drop).values, df_train[target_class].values
    X_test, y_test = df_test.drop(columns=columns_to_drop).values, df_test[target_class].values
    # scale the data
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # encode the classes
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    return X_train, X_test, y_train, y_test, label_encoder


def plot_cm(clf, ax, x_test: np.ndarray, y_test: np.ndarray, labels: list, normalize: str = None) -> None:
    """
    Plots a confusion matrix for the given classifier on the test data.

    Args:
        clf (estimator): An estimator object implementing 'fit' and 'predict'.
        ax (matplotlib.axes.Axes): A Matplotlib Axes object to plot the confusion matrix on.
        x_test (np.ndarray): Array-like object of shape (n_samples, n_features) containing the input features.
        y_test (np.ndarray): Array-like object of shape (n_samples,) containing the target values.
        labels (list): A list of strings representing the class labels.
        normalize (str): Normalization of the confusion matrix

    Returns:
        None.

    Raises:
        ValueError: If the input estimator is not a classifier.

    """
    y_pred = clf.predict(x_test)
    if len(labels) == 2:
        avg = 'binary'
    else:
        avg = 'macro'
    f1 = round(f1_score(y_true=y_test, y_pred=y_pred, average=avg), 2)
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred, normalize=normalize)
    sns.heatmap(cm, annot=True, cmap='Blues', ax=ax, fmt='.2f', square='all',
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(f'{clf.__class__.__name__}\nf1-score: {f1}')


def plot_feature_importances(clf, feature_names: List[str], ax) -> None:
    """
    Plot feature importances for a classifier.

    Args:
        clf : The trained classifier.
        feature_names (list of str): The list of feature names.
        ax (matplotlib.axes.Axes): The axis on which to plot the feature importances.

    Returns:
        None
    """
    # Select the top k features
    k = 10  # set k to the number of top features you want to display
    feature_importances = clf.feature_importances_
    top_k_idx = feature_importances.argsort()[-k:]
    # Plot the top k feature importances
    ax.barh(feature_names[top_k_idx], feature_importances[top_k_idx])
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {k} feature importances")


def pipeline_gridsearch(df: pd.DataFrame, target_class: str = 'culture'):
    start = time.time()
    root_path = os.path.abspath('../')
    # if input(f'Save models in figures in {root_path}/ (y/N)') in ["N", ""]:
    #     root_path = input('Root directory for models and figures: ')
    for dir in ['models', 'fig']:
        dir_path = os.path.join(root_path, dir)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
    if target_class == 'filiere':
        group = 'groups'
    elif df.filiere.unique().shape[0] == 1:
        group = df.filiere.unique()[0]
    else:
        group = 'crops'
    print(f'Processing {group}..')
    X_train, X_test, y_train, y_test, label_encoder = get_xy(df, target_class)
    classes = label_encoder.classes_
    # best classifiers
    classifiers = [SVC, RandomForestClassifier, XGBClassifier]
    param_rf = {'n_estimators': [10, 25, 50, 100],
                'criterion': ['gini', 'entropy'],
                'max_depth': [5, 10, 15, 20, 25, None],
                'min_samples_leaf': [1, 2, 5, 10],
                'random_state': [1]}
    param_svm = {'C': [0.1, 1, 10, 100],
                 'gamma': [1, 0.1, 0.01, 0.001],
                 'kernel': ['rbf', 'poly']}
    param_xgb = {'min_child_weight': [1, 5, 10],
                 'gamma': [0.5, 1, 1.5, 2, 5],
                 'subsample': [0.6, 0.8, 1.0],
                 'colsample_bytree': [0.6, 0.8, 1.0],
                 'max_depth': [3, 4, 5],
                 'random_state': [1]}
    param_grids = [param_svm, param_rf, param_xgb]
    fig, axs = plt.subplots(1, len(classifiers), figsize=(16, 6))
    for classifier, param_grid in zip(classifiers, param_grids):
        if classifier == XGBClassifier:
            clf = custom_tune_v1(x_train=X_train, x_test=X_test,
                                 y_train=y_train, y_test=y_test,
                                 model=classifier, grid_params=param_grid, verbose=False)
        else:
            clf = custom_tune_v2(x_train=X_train, x_test=X_test,
                                 y_train=y_train, y_test=y_test,
                                 model=classifier, grid_params=param_grid, verbose=False)
        model_name = f'models/{group}_{clf.__class__.__name__}.pickle'
        pickle.dump(clf, open(os.path.join(root_path, model_name), 'wb'))
        ax = axs[classifiers.index(classifier)]
        plot_cm(clf=clf, ax=ax, x_test=X_test, y_test=y_test, labels=classes, normalize='true')
        fig_name = f'fig/{group}_cm.png'
        plt.savefig(os.path.join(root_path, fig_name), dpi=150)
        print()
    end = time.time()
    exec_time = int(round(end - start, 0))
    print(f'Done! ({exec_time} seconds)')


def hierarchical_pred(x: np.array, broad_clf, fine_clf, le: Dict[str, LabelEncoder]) -> Tuple:
    """
    Predicts broad and fine classes for given input data using a hierarchical approach.

    Args:
        x: Input array
        broad_clf: The broad classifier used for prediction.
        fine_clf: A dictionary of fine classifiers used for prediction, where the keys are the broad class names.
        le: A dictionary of label encoders , where the keys are the broad class names.

    Returns:
        A tuple containing:
            - broad crop classes
            - fine crop classes
    """
    broad_classes = broad_clf.predict(x.reshape(-1, x.shape[1]))
    fine_classes = np.zeros_like(broad_classes)
    for label in le['groups'].classes_:
        indices = np.where(broad_classes == le['groups'].transform([label]))
        if label == 'oleagineux':
            fine_classes[indices] = le['crops'].transform(['colza'])
        elif label == 'maraicheres':
            fine_classes[indices] = le['crops'].transform(['melon'])
        elif label == 'arboriculture':
            fine_classes[indices] = le['crops'].transform(['agrumes'])
        else:
            initial_labels = fine_clf[label].predict(x[indices])
            str_labels = le[label].inverse_transform(initial_labels)
            fine_classes[indices] = le['crops'].transform(str_labels)
    return broad_classes, fine_classes


def process_combination(args, keys, model, x_train, y_train, x_test, y_test):
    kwargs = dict(zip(keys, args))
    clf = model(**kwargs)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
    return kwargs, f1


def custom_tune_v2(x_train, x_test, y_train, y_test, model, grid_params, verbose: bool = False):
    '''
    Custom grid search function. Uses multiprocessing.
    '''
    start = time.time()
    params = []
    f1_list = []
    model_name = model().__class__.__name__
    keys = list(grid_params.keys())  # Convert keys to a list
    values = grid_params.values()
    num_combinations = 1
    for value_list in values:
        num_combinations *= len(value_list)
    print(f'{time.strftime("%H:%M:%S")} | {model_name} | Fitting for {num_combinations} combinations')
    combinations = itertools.product(*values)

    # Create a pool of worker processes
    pool = multiprocessing.Pool()
    worker_func = partial(process_combination, keys=keys, model=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    # Map the worker function to combinations and collect results
    results = pool.map(worker_func, combinations)

    # Extract params and f1 scores from results
    params, f1_list = zip(*results)

    max_f1_index = f1_list.index(max(f1_list))
    print(20 * '-')
    print(f'Best params: {params[max_f1_index]}')

    execution_time = time.time() - start
    if execution_time > 3600:
        print_time = f'{execution_time / 60 / 60}h'
    elif 60 < execution_time < 3600:
        print_time = f'{execution_time / 60}m'
    elif execution_time < 60:
        print_time = f'{execution_time}s'
    print(f'Max f1 score: {max(f1_list)} ({print_time})')

    best_model = model(**params[max_f1_index])
    best_model.fit(x_train, y_train)

    return best_model


def custom_tune_v1(x_train, x_test, y_train, y_test, model, grid_params, verbose: bool = False):
    '''
    Custom grid search function. Does not use multiprocessing.
    '''
    start = time.time()
    params = []
    f1_list = []
    model_name = model().__class__.__name__
    keys = grid_params.keys()
    values = grid_params.values()
    num_combinations = 1
    for value_list in values:
        num_combinations *= len(value_list)
    print(f'{time.strftime("%H:%M:%S")} | {model_name} | Fitting for {num_combinations} combinations')
    combinations = itertools.product(*values)
    for args in combinations:
        kwargs = dict(zip(keys, args))
        clf = model(**kwargs)
        params.append(kwargs)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        f1_list.append(f1_score(y_true=y_test, y_pred=y_pred, average='macro'))
        if verbose:
            print(f'{model_name} | {kwargs}')
        if verbose:
            print(f'F1-score {f1_score(y_true=y_test, y_pred=y_pred, average="macro")}\n')
    max_f1_index = f1_list.index(max(f1_list))
    print(20 * '-')
    print(f'Best params: {params[max_f1_index]}')
    execution_time = time.time() - start
    if execution_time > 3600:
        print_time = f'{execution_time / 60 / 60}h'
    elif 60 < execution_time < 3600:
        print_time = f'{execution_time / 60}m'
    elif execution_time < 60:
        print_time = f'{execution_time}s'
    print(f'Max f1 score: {max(f1_list)} ({print_time})')
    best_model = model(**params[max_f1_index])
    best_model .fit(x_train, y_train)
    return best_model


def split(geodataframe, target):
    d = {}
    for t in geodataframe[target].unique():
        d[t] = geodataframe[geodataframe[target] == t]
        print(t, d[t].COUNT.sum())
        cond = False
        while not cond:
            subset_1 = d[t].sample(frac=.75)
            subset_2 = d[t].drop(subset_1.index)
            if subset_1.COUNT.sum() > subset_2.COUNT.sum():
                subset_1['TRAIN'] = True
                subset_2['TRAIN'] = False
            else:
                subset_1['TRAIN'] = False
                subset_2['TRAIN'] = True
            d[t] = pd.concat([subset_1, subset_2])
            lim_min = 20 * d[t].COUNT.sum() / 100
            lim_max = 45 * d[t].COUNT.sum() / 100
            test_pixels = d[t].query('TRAIN==False').COUNT.sum()
            if lim_min <= test_pixels <= lim_max:
                cond = True
    full_df = pd.concat([d[i] for i in d])
    return full_df
