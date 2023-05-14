import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
from typing import Tuple, List
import matplotlib.pyplot as plt
# import keras
# from keras.callbacks import EarlyStopping
# from keras.utils import to_categorical


def reflectance_plot(df: pd.DataFrame, column: str) -> None:
    """Create a plot of reflectance values over a range of wavelengths.

    Args:
        df (pandas.DataFrame): A DataFrame containing the reflectance values for different bands and dates.
        column (str): The name of the column in `df` containing the labels for each plot.

    Returns:
        None

    Raises:
        ValueError: If the number of bands in `df` is not a multiple of 10.

    """
    plt.clf()
    dates = ['2020-12-22',
             '2020-12-27',
             '2021-01-03',
             '2021-01-18',
             '2021-01-26',
             '2021-02-15',
             '2021-03-14',
             '2021-03-22',
             '2021-03-24',
             '2021-04-13',
             '2021-04-18',
             '2021-05-06',
             '2021-05-18',
             '2021-05-21'
             ]
    wave = [.490, .560, .665, .705, .740, .783, .842, .865, 1.610, 2.190]
    bands = df.drop(
        columns=['CROP', 'GROUP', 'TRAIN']).columns.tolist()
    labels = df[f'{column}'].unique().tolist()
    H, W = 5, 3
    fig, axs = plt.subplots(H, W, figsize=(W+6, H+6))
    plt.delaxes(axs[H-1, W-1])
    c = 0
    idx = 0
    try:
        while c < len(bands):
            for h in range(H):
                for w in range(W):
                    for label in labels:
                        x = df[df[f'{column}'] == label][bands[c:c+10]]
                        x.columns = wave[0:10]
                        x = x.median()/10_000
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


def ndvi_plot(ndvi: pd.DataFrame, column: str):
    """Plot the NDVI values for each label in a given column over time.

    Args:
        ndvi (pd.DataFrame): A pandas DataFrame containing the NDVI values.
        column (str): The name of the column to group by.

    Returns:
        None
    """
    plt.clf()
    dates = ['2020-12-22',
             '2020-12-27',
             '2021-01-03',
             '2021-01-18',
             '2021-01-26',
             '2021-02-15',
             '2021-03-14',
             '2021-03-22',
             '2021-03-24',
             '2021-04-13',
             '2021-04-18',
             '2021-05-06',
             '2021-05-18',
             '2021-05-21'
             ]
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ndvi[f'{column}'].unique().tolist()
    for label in labels:
        df = ndvi[ndvi[f'{column}'] == label].median(numeric_only=True)
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


def get_xy(df: pd.DataFrame, target_class: str = 'CROP') \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
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
    columns_to_drop = ['CROP', 'GROUP', 'TRAIN']
    df_train, df_test = df.query('TRAIN==True'), df.query('TRAIN==False')
    X_train, y_train = df_train.drop(
        columns=columns_to_drop).values, df_train[target_class].values
    X_test, y_test = df_test.drop(
        columns=columns_to_drop).values, df_test[target_class].values

    # scale the data
    # X_train, X_test = X_train / 10_000, X_test / 10_000
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # encode the classes
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    return X_train, X_test, y_train, y_test, label_encoder


def cnet_preprocess(df: pd.DataFrame):
    non_bands_cols = ['CROP', 'GROUP', 'TRAIN']
    # calculate NDVI
    NDVI = df[non_bands_cols].copy()
    for i in range(14):
        if i == 0:
            NDVI.loc[:, f'V{i + 1}'] = (df['B7'] -
                                        df['B3']) / (df['B7'] + df['B3'])
        else:
            NDVI.loc[:, f'V{i + 1}'] = (df[f'B{i}7'] -
                                        df[f'B{i}3']) / (df[f'B{i}7'] + df[f'B{i}3'])
    NDVI = NDVI.dropna()
    df = df.loc[NDVI.index]
    #
    # X: CNN dataset
    df_train_B, df_test_B = df.query(
        'TRAIN == True'), df.query('TRAIN == False')
    X_train_B = df_train_B.drop(columns=non_bands_cols).values
    X_test_B = df_test_B.drop(columns=non_bands_cols).values
    X_train_B, X_test_B = X_train_B.reshape(-1, 14, 10) / \
        10_000, X_test_B.reshape(-1, 14, 10) / 10_000
    # X: lstm dataset
    df_train_VI, df_test_VI = NDVI.query(
        'TRAIN == True'), NDVI.query('TRAIN == False')
    X_train_VI = df_train_VI.drop(columns=non_bands_cols).values
    X_test_VI = df_test_VI.drop(columns=non_bands_cols).values
    X_train_VI, X_test_VI = X_train_VI.reshape(
        -1, 14, 1) / 10_000, X_test_VI.reshape(-1, 14, 1) / 10_000
    # y
    label_encoder = LabelEncoder()
    y_train, y_test = df_train_VI['CROP'].values, df_test_VI['CROP'].values
    y_train = to_categorical(label_encoder.fit_transform(y_train))
    y_test = to_categorical(label_encoder.transform(y_test))
    return X_train_B, X_train_VI, X_test_B, X_test_VI, y_train, y_test, label_encoder


def cnet_model(X_train_B, X_train_VI, X_test_B, X_test_VI, y_train, y_test, label_encoder):
    # cnn branch
    inputs_cnn = keras.layers.Input(X_train_B.shape[1:])
    branch_cnn = keras.layers.Conv1D(
        filters=8, kernel_size=1, activation='relu')(inputs_cnn)
    branch_cnn = keras.layers.MaxPooling1D(pool_size=3)(branch_cnn)
    # lstm branch
    inputs_lstm = keras.layers.Input(X_train_VI.shape[1:])
    branch_lstm = keras.layers.LSTM(
        units=8, return_sequences=True)(inputs_lstm)
    # merge branches
    X = keras.layers.concatenate([branch_cnn, branch_lstm], axis=1)
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(units=512, activation='relu')(X)
    X = keras.layers.Dropout(0.4)(X)
    X = keras.layers.Dense(units=128, activation='relu')(X)
    X = keras.layers.Dense(units=32, activation='relu')(X)
    classes = label_encoder.classes_
    num_classes = len(classes)
    outputs = keras.layers.Dense(units=num_classes, activation='softmax')(X)
    model = keras.Model(inputs=[inputs_cnn, inputs_lstm], outputs=outputs)
    optimizer = keras.optimizers.Adam(
        learning_rate=.001, decay=.005, beta_1=.8, beta_2=.9)
    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       patience=10,
                       restore_best_weights=True)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    model.fit(x=[X_train_B, X_train_VI],
              y=y_train,
              callbacks=es,
              epochs=500,
              batch_size=512,
              validation_data=([X_test_B, X_test_VI], y_test),
              verbose=0)
    return model


def get_best_model(clf, param_grid: dict, x_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Trains and tunes a classification model using grid search cross-validation.

    Args:
        clf (estimator): An estimator object implementing 'fit' and 'predict'.
        param_grid (dict): A dictionary with parameters names (string) as keys and lists of parameter settings to try
            as values. The parameter settings are combinations of values to be tested by the randomized search.
        x_train (np.ndarray): Array-like object of shape (n_samples, n_features) containing the input features.
        y_train (np.ndarray): Array-like object of shape (n_samples,) containing the target values.

    Returns:
        tuple: A tuple containing the best estimator and its corresponding parameters, as determined by the randomized
         search.

    Raises:
        ValueError: If the input estimator is not a classifier.

    """
    model = GridSearchCV(estimator=clf,
                         param_grid=param_grid,
                         refit=True,
                         cv=3,
                         n_jobs=-1,
                         verbose=0)
    model.fit(x_train, y_train)
    best_model = model.best_estimator_
    best_params = model.best_params_
    return best_model, best_params


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
        avg = 'weighted'
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
