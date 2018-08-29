
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#loading the file
passengers = pd.read_csv('air_passengers.csv', index_col='Month')
passengers.index = pd.to_datetime(passengers.index, format = '%Y-%m')
passengers.head()


#plot the entire timeseries
plt.figure(figsize=(12, 4))
plt.plot(passengers)
plt.xlabel('Years')
plt.ylabel('Passengers')
plt.show()


#function to compute the rolling mean and std. dev.
def rolling_stats(timeseries):
    #Determine rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    plt.figure(figsize=(12, 4))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=True)

rolling_stats(passengers['count'])


#dickey-fuller test
from statsmodels.tsa.stattools import adfuller
def adf(timeseries): 
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags                     Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

adf(passengers['count'])


#taking the log transform
pax_log = np.log(passengers)
plt.figure(figsize=(12, 4))
plt.plot(pax_log)
plt.xlabel('Years')
plt.ylabel('Log transformed count of passengers')
plt.show()


#differencing
pax_log_diff = pax_log.diff()
plt.figure(figsize=(12, 4))
plt.plot(pax_log_diff)
plt.show()


#computing the rolling stats and dickey fuller test
pax_log_diff.dropna(inplace=True)
rolling_stats(pax_log_diff)
adf(pax_log_diff['count'])



#partitioning the data into training and test set 
split_ratio = .75
split_point = int(round(len(pax_log) * split_ratio))
train, test = pax_log.iloc[:split_point,:], pax_log.iloc[split_point:,:]
train_diff = train.diff()
train_diff.dropna(inplace=True)



#plotting acf and pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(train_diff, lags=20, alpha = 0.05)
plot_pacf(train_diff, lags=20, alpha = 0.05)


#creating a seasonal arima model
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(train, order=(4, 1, 4), seasonal_order = (1,0,0,12), enforce_stationarity=False, enforce_invertibility=False ) 
model_fit= model.fit(disp=0) 


#generating the forecasts
K = len(test)
forecast = model_fit.forecast(K)
forecast = np.exp(forecast)

plt.figure(figsize=(12, 4))
plt.plot(forecast, 'r')
plt.plot(passengers, 'b')
plt.title('RMSE: %.4f'% np.sqrt(sum((forecast - np.exp(test['count']))**2)/len(test)))
plt.show()