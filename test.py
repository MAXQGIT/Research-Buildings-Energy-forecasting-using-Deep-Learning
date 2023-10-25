import pandas as pd

data = pd.read_csv('content/preprocessed_train.csv')
print(type(data))
building_id = 122
meter = 0
primary_use = 9
df = data.query(f'building_id=={building_id} & meter=={meter}')
print(type(df))
new_df = df.drop(['building_id', 'meter', 'primary_use'], axis=1)
print(new_df)
print(type(new_df))
