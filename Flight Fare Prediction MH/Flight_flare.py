import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm
from sklearn.metrics import r2_score, mean_squared_error

# %%
# Preprocessing
df = pd.read_excel('Data_train.xlsx', engine='openpyxl')
df = df.dropna()
df = df.drop(columns=['Additional_Info', 'Route'], axis=1)
df['Dep_H'] = df.apply(lambda x: int(x['Dep_Time'].split(':')[0]), axis=1)
df['Dep_M'] = df.apply(lambda x: int(x['Dep_Time'].split(':')[1]), axis=1)
df['Arrival_Time_H'] = df.apply(lambda x: int(x['Arrival_Time'].split(':')[0]), axis=1)
df['Arrival_Time_M'] = df.apply(lambda x: int(x['Arrival_Time'].split(':')[1].split(' ')[0]), axis=1)
df['Duration_H'] = df['Duration'].apply(lambda x: int(x.split('h')[0]) if 'h' in x else 0)
df['Duration_M'] = df['Duration'].apply(lambda x: 0 if 'm' not in x else int(x.split('h')[1].split('m')[0]) if 'h' in x else 0)
df = df.drop(columns=['Dep_Time', 'Arrival_Time', 'Duration'], axis=1)
df['date_M'] = df.apply(lambda x: int(x['Date_of_Journey'].split('/')[1]), axis=1)
df['date_D'] = df.apply(lambda x: int(x['Date_of_Journey'].split('/')[0]), axis=1)
df = df.drop('Date_of_Journey', axis=1)
df = df.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4})
# df = df.replace(dict(zip(df['Airline'].unique(), range(12))))
# df = df.replace(dict(zip(df['Source'].unique(), range(5))))
# df = df.replace(dict(zip(df['Destination'].unique(), range(6))))
# data_train = df
df = df.replace(['Trujet', 'Vistara Premium economy', 'Jet Airways Business', 'Multiple carriers Premium economy'], 'Other')
data_train = pd.get_dummies(df, drop_first=True, columns=['Source', 'Airline', 'Destination'])

# %%
# Training
X = data_train.drop('Price', axis=1)
y = data_train['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#%%
train_data = lightgbm.Dataset(X_train, y_train)
val_data = lightgbm.Dataset(X_test, y_test, reference=train_data)
#%%
params = {'objective': 'tweedie',
          'metric': ['Huber', 'rmse'],
          'learning_rate': 0.1,
          'bagging_fraction': 0.8,
          'bagging_freq': 5,
          'feature_fraction': 0.9,
          'min_data_in_leaf': 20,
          'num_leaves': 41,
          'scale_pos_weight': 1.2,
          'lambda_l2': 1,
          'lambda_l1': 1
          }
gbm = lightgbm.train(params=params,
                     train_set=train_data,
                     valid_sets=[train_data, val_data],
                     valid_names=['train', 'valid'])
# %%
print(r2_score(y_test, gbm.predict(X_test)), mean_squared_error(y_test, gbm.predict(X_test)))
# %%
path = './lgbm_flight_flare.h5'
gbm.save_model(path)
