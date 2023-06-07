import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from ml_utils import get_xy, hierarchical_pred
from keras.models import load_model
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

df = pd.read_parquet('../data/culture_dataset.parquet')
df = df.drop(df.query('culture=="avoine"').index)

le = {}
_, X, _, y, le['crops'] = get_xy(df=df, target_class='culture')
fine_labels = le['crops'].classes_

*_, le['groups'] = get_xy(df=df, target_class='filiere')
classes = le['groups'].classes_
clf_groups = pickle.load(open('../models/groups_XGBClassifier.pickle', 'rb'))

fine_classifiers = {}
for f in ['fruitiers', 'cereales', 'legumineuses']:
    *_, le[f] = get_xy(df=df.query('filiere==@f'), target_class='culture')

fine_classifiers['fruitiers'] = pickle.load(open('../models/fruitiers_SVC.pickle', 'rb'))
fine_classifiers['legumineuses'] = pickle.load(open('../models/legumineuses_RandomForestClassifier.pickle', 'rb'))

cnet = True
if cnet:
    fine_classifiers['cereales'] = load_model('../models/cereales_dl.h5')
else:
    fine_classifiers['cereales'] = pickle.load(open('../models/cereales_SVC.pickle', 'rb'))

y_pred_broad, y_pred_fine = hierarchical_pred(X, clf_groups, fine_classifiers, le, cnet)

fig, ax = plt.subplots(figsize=(8, 6))
f1 = round(f1_score(y_true=y, y_pred=y_pred_fine, average='macro'), 2)
cm = confusion_matrix(y_true=y, y_pred=y_pred_fine, normalize='true')
sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', square='all',
            xticklabels=fine_labels, yticklabels=fine_labels)
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.set(title=f'C-Net: {cnet}, F1-score: {f1}')
plt.savefig('../fig/hierarchical_cm.png', dpi=150)

print(classification_report(y_true=y, y_pred=y_pred_fine, target_names=fine_labels))
