import pandas as pd
from datetime import datetime

df = pd.read_csv('sphist.csv')
df['Date'] = pd.to_datetime(df.Date)
df_ordered = df.sort('DateTime', ascending=True).reset_index()

row_to_drop = df_ordered["Date"] < datetime(year=2015, month=4, day=1)
df_ordered = df_ordered.drop(row_to_drop, axis=0)

data_mean_5day = pd.rolling_mean(df_ordered.Close, window=5).shift(1)
data_mean_365day = pd.rolling_mean(df_ordered.Close, window=365).shift(1)
data_mean_ratio = data_mean_5day/data_mean_365day

data_std_5day = pd.rolling_std(df_ordered.Close, window=5).shift(1)
data_std_365day = pd.rolling_std(df_ordered.Close, window=365).shift(1)
data_std_ratio = data_std_5day/data_std_365day

df_ordered['data_mean_5day'] = data_mean_5day
df_ordered['data_mean_365day'] = data_mean_365day
df_ordered['data_mean_ratio'] = data_mean_ratio
df_ordered['data_std_5day'] = data_std_5day
df_ordered['data_std_365day'] = data_std_365day
df_ordered['data_std_ratio'] = data_std_ratio

# remove rows before 1951-01-03 that don't have enough data
df_new = df_ordered[df_ordered["Date"] > datetime(year=1951, month=1, day=2)]

# remove any rows with NaN
df_new.dropna(axis=0, inplace=)

# train: data less than 2013-01-01
# test: data greater than 2013-01-01
train = df_new[df_new["Date"] < datetime(year=2013, month=1, day=1)]
test = df_new.loc[~df_new.index.isin(train.index)]


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

to_drop = ['Close', 'High', 'Low', 'Open', 'Volume', 'Adj', 'Close', 'Date']
features = df_new.columns.drop(to_drop)

lr = LinearRegression()
lr.fit(train[features], train['Close'])
prediction = lr.predict(test[features])
mse = mean_squared_error(test['Close'], prediction)


