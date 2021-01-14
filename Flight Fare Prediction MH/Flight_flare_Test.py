import pandas as pd
import lightgbm

#%%
# Preprocessing
df = pd.read_excel('Test_set.xlsx', engine='openpyxl')
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
data_test = pd.get_dummies(df, drop_first=True, columns=['Source', 'Airline', 'Destination'])
#%%
X_test = data_test
path = './lgbm_flight_flare.h5'
gbm = lightgbm.Booster(model_file=path)
#%%
gbm.predict(X_test)
