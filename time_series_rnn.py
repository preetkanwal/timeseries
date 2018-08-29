
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LSTM 


passengers = pd.read_csv('air_passengers.csv', index_col='Month')
passengers.index = pd.to_datetime(passengers.index, format = '%Y-%m')


#define adjusted r-squared
def adj_r2_score(r2, n, k):
    return 1-((1-r2)*((n-1)/(n-k-1)))


#Partition data into training and test set
split_ratio = .75
split_point = int(round(len(passengers) * split_ratio))
train, test = passengers.iloc[:split_point,:], passengers.iloc[split_point:,:]


#pre-processing the input data using min-max scaling
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)


#preparing the training dataset
X_tr_sc = train_scaled[:-1]
y_tr_sc = train_scaled[1:]
X_tst_sc = test_scaled[:-1]
y_tst_sc = test_scaled[1:]

X_tr_sc = X_tr_sc.reshape(X_tr_sc.shape[0], 1, X_tr_sc.shape[1])
X_tst_sc = X_tst_sc.reshape(X_tst_sc.shape[0], 1, X_tst_sc.shape[1])


#training the LSTM model
K.clear_session()
model_lstm = Sequential()
model_lstm.add(LSTM(7, input_shape=(1, X_tr_sc.shape[1]), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
history_model_lstm = model_lstm.fit(X_tr_sc, y_tr_sc, epochs=50, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])

test_pred_sc = model_lstm.predict(X_tst_sc)
train_pred_sc = model_lstm.predict(X_tr_sc)

print("R2 score on the Train set:\t{:0.3f}".format(r2_score(y_tr_sc, train_pred_sc)))
r2_train = r2_score(y_tr_sc, train_pred_sc)

print("Adjusted R2 score on the Train set:\t{:0.3f}\n".format(adj_r2_score(r2_train, X_tr_sc.shape[0], X_tr_sc.shape[1])))

print("R2 score on the Test set:\t{:0.3f}".format(r2_score(y_tst_sc, test_pred_sc)))
r2_test = r2_score(y_tst_sc, test_pred_sc)

print("Adjusted R2 score on the Test set:\t{:0.3f}".format(adj_r2_score(r2_test, X_tst_sc.shape[0], X_tst_sc.shape[1])))

score_lstm= model_lstm.evaluate(X_tst_sc, y_tst_sc, batch_size=1)
print('LSTM: %f'%score_lstm)


#making predictions on the test set
test_pred_sc = model_lstm.predict(X_tst_sc)

plt.plot(y_tst_sc, label='Actual')
plt.plot(test_pred_sc, label='Predicted')
plt.title("LSTM's_Prediction (scaled)")
plt.xlabel('Observation')
plt.ylabel('Scaled values')
plt.legend()
plt.show()


#plotting predictions and actuals on the original scale
test_pred = scaler.inverse_transform(test_pred_sc)
test_pred = test_pred.ravel()
test_pred = pd.Series(test_pred)
test_pred.index = test.index[1:]
y_test = test[1:]
plt.figure(figsize=(12, 4))
plt.plot(passengers, 'b')
plt.plot(test_pred, 'r')

plt.title('RMSE: %.4f'% np.sqrt(sum((test_pred.values - y_test['count'].values)**2)/len(y_test)))

