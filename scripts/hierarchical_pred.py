from ml_utils import get_xy
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from typing import Any, List, Dict, Tuple
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

df = pd.read_parquet('../data/gharb_2021_pts.parquet')

# # this is to insure that no true divide errors occure
# non_bands_cols = ['CROP', 'GROUP', 'TRAIN']
# # calculate NDVI
# NDVI = df[non_bands_cols].copy()
# for i in range(14):
#     if i == 0:
#         NDVI.loc[:, f'V{i + 1}'] = (df['B7'] - df['B3']) / (df['B7'] + df['B3'])
#     else:
#         NDVI.loc[:, f'V{i + 1}'] = (df[f'B{i}7'] - df[f'B{i}3']) / (df[f'B{i}7'] + df[f'B{i}3'])
# NDVI = NDVI.dropna()
# df = df.loc[NDVI.index]

_, X, _, y, crops_le = get_xy(df=df, target_class='CROP')

_, _, _, _, broad_le = get_xy(df=df, target_class='GROUP')
classes = broad_le.classes_
clf_groups = pickle.load(open('../models/groups_SVC.pickle', 'rb'))

# fruits
df_fruits = df.query('GROUP=="Fruits"').dropna()
_, _, _, _, fruits_le = get_xy(df=df_fruits, target_class='CROP')
clf_fruits = pickle.load(open('../models/Fruits_RandomForestClassifier.pickle', 'rb'))

# leguminous
df_legum = df.query('GROUP=="Leguminous"').dropna()
_, _, _, _, legum_le = get_xy(df=df_legum, target_class='CROP')
clf_legum = pickle.load(open('../models/Leguminous_SVC.pickle', 'rb'))

# cereals
df_cereal = df.query('GROUP=="Cereals"').dropna()
_, _, _, _, cereal_le = get_xy(df=df_cereal, target_class='CROP')
clf_cereal = pickle.load(open('../models/Cereals_RandomForestClassifier.pickle', 'rb'))


fine_classifiers = {
    'Fruits': clf_fruits,
    'Leguminous': clf_legum,
    'Cereals': clf_cereal
}

label_encoders = {
    'Fruits': fruits_le,
    'Leguminous': legum_le,
    'Cereals': cereal_le
}


def hierarchical_pred(x: np.array,
                      broad_classifier: Any,
                      fine_classifiers: Dict[str, Any],
                      label_encoders: Dict[str, LabelEncoder]
                      ) -> Tuple[List[str], List[str]]:
    """
    Predicts broad and fine classes for given input data using a hierarchical approach.

    Args:
        x: Input array
        broad_classifier: The broad classifier used for prediction.
        fine_classifiers: A dictionary of fine classifiers used for prediction, where the keys are the broad class names.
        label_encoders: A dictionary of label encoders , where the keys are the broad class names.

    Returns:
        A tuple containing:
            - List of broad crop classes
            - List of fine crop classes
    """
    broad_classes = []
    fine_classes = []
    for elem in x:
        broad_class = broad_classifier.predict(elem.reshape(-1, 140))
        broad_class = broad_le.inverse_transform(broad_class)[0]
        if broad_class == 'Oilseed':
            fine_class = 'Rapeseed'
        elif broad_class == 'Vegetables and melons':
            fine_class = 'Melon'
        # elif broad_class == 'Cereals':
        #     x_B = x.reshape(-1, 14, 10)
        #     x_VI = (x_B[:, :, 6] - x_B[:, :, 2]) / (x_B[:, :, 6] + x_B[:, :, 2])
        #     fine_class = fine_classifiers[broad_class].predict([x_B, x_VI], verbose=0)
        else:
            fine_class = fine_classifiers[broad_class].predict(elem.reshape(-1, 140))
            fine_class = label_encoders[broad_class].inverse_transform(fine_class)[0]

        broad_classes.append(broad_class)
        fine_classes.append(fine_class)

    return broad_classes, fine_classes


y_pred_broad, y_pred_fine = hierarchical_pred(X, clf_groups, fine_classifiers, label_encoders)

y_pred = crops_le.transform(y_pred_fine)
labels = crops_le.classes_

print(classification_report(y_true=y, y_pred=y_pred, target_names=labels))


fig, ax = plt.subplots(figsize=(16, 6))
f1 = round(f1_score(y_true=y, y_pred=y_pred, average='weighted'), 2)
cm = confusion_matrix(y_true=y, y_pred=y_pred, normalize='true')
sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', square='all',
            xticklabels=labels, yticklabels=labels)
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.set(title=f'F1-score: {f1}')
plt.show()
