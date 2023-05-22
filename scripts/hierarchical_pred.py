from ml_utils import get_xy, hierarchical_pred
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

df = pd.read_parquet('../data/ee_sampled_pts_df_2021.parquet')

_, X, _, y, crops_le = get_xy(df=df, target_class='culture')

*_, broad_le = get_xy(df=df, target_class='filiere')
classes = broad_le.classes_
clf_groups = pickle.load(open('../models/groups_SVC.pickle', 'rb'))

label_encoders = {}
fine_classifiers = {}
for f in ['arboriculture', 'cereales', 'legumineuses', 'maraicheres']:
    *_, label_encoders[f] = get_xy(df=df.query('filiere==@f'), target_class='culture')
    fine_classifiers[f] = pickle.load(open(f'../models/{f}_RandomForestClassifier.pickle', 'rb'))

label_encoders['groups'] = broad_le

y_pred_broad, y_pred_fine = hierarchical_pred(X, clf_groups, fine_classifiers, label_encoders)
y_pred = crops_le.transform(y_pred_fine)
labels = crops_le.classes_

fig, ax = plt.subplots(figsize=(8, 6))
f1 = round(f1_score(y_true=y, y_pred=y_pred, average='weighted'), 2)
cm = confusion_matrix(y_true=y, y_pred=y_pred, normalize='true')
sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', square='all',
            xticklabels=labels, yticklabels=labels)
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.set(title=f'F1-score: {f1}')
plt.savefig('../fig/hierarchical_cm.png', dpi=150)

print(classification_report(y_true=y, y_pred=y_pred, target_names=labels))
