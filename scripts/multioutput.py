import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, RobustScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import time

start = time.time()
# tf.random.set_seed(99)
target_class: str = 'filiere'
df = pd.read_parquet('../data/ee_sampled_pts_df_2021.parquet')

bands = df.drop(columns=['TRAIN', 'culture', 'filiere']).columns
df_train, df_test = df.query('TRAIN==True'), df.query('TRAIN==False')
X_train, y_train = df_train[bands].values, df_train[['filiere', 'culture']].values
X_test, y_test = df_test[bands].values, df_test[['filiere', 'culture']].values

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# mlb = MultiLabelBinarizer()
# y_train = mlb.fit_transform(y_train)
# y_test = mlb.transform(y_test)

# forest = RandomForestClassifier(random_state=1)
# multi_target_forest = MultiOutputClassifier(forest, n_jobs=2)
# multi_target_forest.fit(X_train, y_train)

# pickle.dump(multi_target_forest, open('../models/multi_target_forest.pickle', 'wb'))

multi_target_forest = pickle.load(open('../models/multi_target_forest.pickle', 'rb'))

y_pred = multi_target_forest.predict(X_test)

# print(f'Execution time: {time.time()-start} seconds.')
