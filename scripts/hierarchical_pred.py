from ml_utils import get_xy, hierarchical_pred
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

df = pd.read_parquet('../data/culture_dataset.parquet')
df = df.drop(df.query('culture=="avoine"').index)

_, X, _, y, crops_le = get_xy(df=df, target_class='culture')

*_, broad_le = get_xy(df=df, target_class='filiere')
classes = broad_le.classes_
clf_groups = pickle.load(open('../models/groups_XGBClassifier.pickle', 'rb'))

label_encoders = {}
fine_classifiers = {}
for f in ['fruitiers', 'cereales', 'legumineuses']:
    *_, label_encoders[f] = get_xy(df=df.query('filiere==@f'), target_class='culture')

fine_classifiers['fruitiers'] = pickle.load(open('../models/fruitiers_SVC.pickle', 'rb'))
fine_classifiers['cereales'] = pickle.load(open('../models/cereales_SVC.pickle', 'rb'))
fine_classifiers['legumineuses'] = pickle.load(open('../models/legumineuses_RandomForestClassifier.pickle', 'rb'))

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
