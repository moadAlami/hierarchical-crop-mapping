import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from ml_utils import get_xy, hierarchical_pred
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

df = pd.read_parquet('../data/culture_dataset_v2.parquet')
df = df.dropna()
to_drop = ['olivier', 'grenadier']
df = df.drop(df.query('culture.isin(@to_drop)').index)

le = {}
_, X, _, y, le['crops'] = get_xy(df=df, target_class='culture')
fine_labels = le['crops'].classes_

*_, le['groups'] = get_xy(df=df, target_class='filiere')
classes = le['groups'].classes_
clf_groups = pickle.load(open('../models/groups_SVC.pickle', 'rb'))

groups = []
for group in df.filiere.unique():
    crops = df.query('filiere==@group').culture.unique()
    if len(crops) > 1:
        groups.append(group)

fine_classifiers = {}
for f in groups:
    *_, le[f] = get_xy(df=df.query('filiere==@f'), target_class='culture')

fine_classifiers['cereales'] = pickle.load(open('../models/cereales_SVC.pickle', 'rb'))
fine_classifiers['legumineux'] = pickle.load(open('../models/legumineux_SVC.pickle', 'rb'))

y_pred_broad, y_pred_fine = hierarchical_pred(X, clf_groups, fine_classifiers, le)

fig, ax = plt.subplots(figsize=(8, 6))
f1 = round(f1_score(y_true=y, y_pred=y_pred_fine, average='macro'), 2)
cm = confusion_matrix(y_true=y, y_pred=y_pred_fine, normalize='true')
sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', square='all',
            xticklabels=fine_labels, yticklabels=fine_labels)
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.set(title=f'F1-score: {f1}')
plt.savefig('../fig/hierarchical_cm.png', dpi=150)

print(classification_report(y_true=y, y_pred=y_pred_fine, target_names=fine_labels))
