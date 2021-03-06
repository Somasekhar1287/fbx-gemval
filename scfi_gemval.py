import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.formula.api as smf
import statsmodels.api as sm



from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from keras.callbacks import EarlyStopping

def scfi(df):
    #scfi=pd.read_csv(df,parse_dates = ['date'], index_col = ['date'])
    #Creating Train set
    scfi_train_w    = df[0:620]
    #Creating Test set
    scfi_test_w     = df[620:627]
    scfi_train_w.describe()
    scfi_train_w.skew()
    scfi_train_w.kurt()
    #creating log for traina nd test
    scfi_train_w_log   = np.log(scfi_train_w)
    scfi_test_w_log     =np.log(scfi_test_w) 
    return scfi,scfi_train_w,scfi_test_w,scfi_train_w_log,scfi_test_w_log
    
def scfi_arima_7days(scfi_train_w_log,scfi_test_w):
    scfi_train_w =scfi_train_w_log.reset_index()
    y_scfi_w = scfi_train_w['value']
    x_scfi_w = range(0, 620)
    scfi_train_w.columns.values
    scfi_w = smf.ols('y_scfi_w ~ x_scfi_w', data =scfi_train_w)
    scfi_w=  scfi_w.fit()
    scfi_w.summary()   
    scfi_stationary_w = y_scfi_w - scfi_w.fittedvalues
    scfi_ar1_w = ARIMA(y_scfi_w, order = (1, 0, 0))
    scfi_ar1_w = scfi_ar1_w.fit()
    scfi_ar1_pred_w = scfi_ar1_w.predict(start = 620, end = 627)
    scfi_ar1_2_w = SARIMAX(y_scfi_w, order = (1, 0, 0))
    scfi_ar1_2_w = scfi_ar1_2_w.fit()
    scfi_ar1_2_w.summary() 
    
    y_pred_scfi_w = scfi_ar1_2_w.get_forecast(len(scfi_test_w.index)+1)
    y_pred_df_scfi_w = y_pred_scfi_w.conf_int(alpha = 0.05) 
    y_pred_df_scfi_w["Predictions"] = scfi_ar1_2_w.predict(start = y_pred_df_scfi_w.index[0], end = y_pred_df_scfi_w.index[-1])
    scfi_ar_pred_w = pd.DataFrame(scfi_ar1_pred_w)
    scfi_test_w['pred_1'] = scfi_ar_pred_w.values[1:8]
    rmspe1_scfi_1_w = np.sqrt(np.sum(scfi_test_w.iloc[:, 0].subtract(scfi_test_w.iloc[:, 1])**2)/7)
    
    y_pred_df_scfi_w.columns.values
    pred_2_w = pd.DataFrame(y_pred_df_scfi_w['Predictions'])
    scfi_test_w['pred_2_w'] = pred_2_w.values[1:8]
    rmspe1_scfi_2_w = np.sqrt(np.sum(scfi_test_w.iloc[:, 0].subtract(scfi_test_w.iloc[:, 1])**2)/7)
    scfi_ar1_w2 = ARIMA(y_scfi_w, order = (1, 0, 1))
    scfi_ar1_w2 = scfi_ar1_w2.fit()
    scfi_ar1_pred_w2 = scfi_ar1_w2.predict(start = 620, end = 627)
    scfi_ar1_2_w2 = SARIMAX(y_scfi_w, order = (1, 1, 0)) #(1 0 1) des not become stationary
    scfi_ar1_2_w2 = scfi_ar1_2_w2.fit()
    y_pred_scfi_w2 = scfi_ar1_2_w2.get_forecast(len(scfi_test_w.index)+1)
    y_pred_df_scfi_w2 = y_pred_scfi_w2.conf_int(alpha = 0.05) 
    y_pred_df_scfi_w2["Predictions"] = scfi_ar1_2_w2.predict(start = y_pred_df_scfi_w2.index[0], end = y_pred_df_scfi_w2.index[-1])
    
    #fig = plt.figure(figsize = (16,8))
    #ax1 = fig.add_subplot(1, 1, 1)
    #plt.plot(y_scfi_w)
    #plt.plot(y_pred_df_scfi_w2["Predictions"],color = 'red',label='predicted')
    #y_pred_df_scfi_w2.head()
    #plt.plot(y_pred_df_scfi_w2['lower value'], linestyle = '--', color = 'red', linewidth = 0.5, label='lower ci')
    #plt.plot(y_pred_df_scfi_w2['upper value'], linestyle = '--', color = 'red', linewidth = 0.5, label='upper ci')
    #plt.fill_between(y_pred_df_scfi_w2["Predictions"].index.values,
                     #y_pred_df_scfi_w2['lower value'], 
                   #  y_pred_df_scfi_w2['upper value'], 
                   #  color = 'grey', alpha = 0.2)
    #plt.legend(loc = 'best')
    # plt.show()
    scfi_ar_pred_w2 = pd.DataFrame(scfi_ar1_pred_w2)
    scfi_test_w['pred_1'] = scfi_ar_pred_w2.values[1:8]
    rmspe2_scfi_1_w = np.sqrt(np.sum(scfi_test_w.iloc[:, 0].subtract(scfi_test_w.iloc[:, 1])**2)/7)

    y_pred_df_scfi_w2.columns.values
    pred_2_w2 = pd.DataFrame(y_pred_df_scfi_w2['Predictions'])
    scfi_test_w['pred_2_w'] = pred_2_w2.values[1:8]
    rmspe2_scfi_2_w = np.sqrt(np.sum(scfi_test_w.iloc[:, 0].subtract(scfi_test_w.iloc[:, 2])**2)/7)
    return y_scfi_w,y_pred_df_scfi_w2,scfi_ar1_2_w2.summary()
    
def scfi_arima_30days(df):
    #ARIMA 30 DAYS
    scfi_Train_m    = df[0:597]
    scfi_Test_m     = df[597:627]
    scfi_Train_m    = np.log(scfi_Train_m)
    scfi_Test_m    = np.log(scfi_Test_m)
    

    scfi_Train_m = scfi_Train_m.reset_index()
    y_scfi_m = scfi_Train_m['value']
    x_scfi_m = range(0, 597)
    scfi_lm_m = smf.ols('y_scfi_m ~ x_scfi_m', data = scfi_Train_m)
    scfi_lm_m = scfi_lm_m.fit()
    scfi_lm_m.summary()
   
    scfi_stationary_m = y_scfi_m - scfi_lm_m.fittedvalues
    

    scfi_ar_m = ARIMA(y_scfi_m, order = (1, 0, 0))
    scfi_ar_m = scfi_ar_m.fit()

    

    scfi_ar_pred_m = scfi_ar_m.predict(start = 597, end = 627)

   
    scfi_ar_2_m = SARIMAX(y_scfi_m, order = (1, 0, 0))

    scfi_ar_2_m = scfi_ar_2_m.fit()
    y_pred_scfi_m = scfi_ar_2_m.get_forecast(len(scfi_Test_m.index)+1)
    y_pred_df_scfi_m = y_pred_scfi_m.conf_int(alpha = 0.05) 
    y_pred_df_scfi_m["Predictions"] = scfi_ar_2_m.predict(start = y_pred_df_scfi_m.index[0], end = y_pred_df_scfi_m.index[-1])
    
    scfi_ar_pred_m = pd.DataFrame(scfi_ar_pred_m)
    scfi_Test_m['pred_1'] = scfi_ar_pred_m.values[1:32]
    rmspe_scfi_1_m = np.sqrt(np.sum(scfi_Test_m.iloc[:, 0].subtract(scfi_Test_m.iloc[:, 1])**2)/30)

    pred_2_m = pd.DataFrame(y_pred_df_scfi_m['Predictions'])
    scfi_Test_m['pred_2_m'] = pred_2_m.values[1:32]
    rmspe__scfi_2_m = np.sqrt(np.sum(scfi_Test_m.iloc[:, 0].subtract(scfi_Test_m.iloc[:, 2])**2)/30)

    #Arima(1 0 1)

    scfi_ar1_m2 = ARIMA(y_scfi_m, order = (1, 0, 1))
    scfi_ar1_m2 = scfi_ar1_m2.fit()

     

    scfi_ar1_pred_m2 = scfi_ar1_m2.predict(start = 597, end = 627)
    

    scfi_ar1_2_m2 = SARIMAX(y_scfi_m, order = (1, 1, 0)) #(1 0 1) des not become stationary
    scfi_ar1_2_m2 = scfi_ar1_2_m2.fit()
    y_pred_scfi_m2 = scfi_ar1_2_m2.get_forecast(len(scfi_Test_m.index)+1)
    y_pred_df_scfi_m2 = y_pred_scfi_m2.conf_int(alpha = 0.05) 
    y_pred_df_scfi_m2["Predictions"] = scfi_ar1_2_m2.predict(start = y_pred_df_scfi_m2.index[0], end = y_pred_df_scfi_m2.index[-1])
    
    scfi_ar_pred_m2 = pd.DataFrame(scfi_ar1_pred_m2)
    scfi_Test_m['pred_1'] = scfi_ar_pred_m2.values[1:32]
    rmspe2_scfi_1_m = np.sqrt(np.sum(scfi_Test_m.iloc[:, 0].subtract(scfi_Test_m.iloc[:, 1])**2)/29)

    y_pred_df_scfi_m2.columns.values
    pred_2_m2 = pd.DataFrame(y_pred_df_scfi_m2['Predictions'])
    scfi_Test_m['pred_2_m'] = pred_2_m2.values[1:32]
    rmspe2_scfi_2_m = np.sqrt(np.sum(scfi_Test_m.iloc[:, 0].subtract(scfi_Test_m.iloc[:, 2])**2)/29)
    return y_scfi_m,y_pred_df_scfi_m,scfi_ar1_m2.summary()
    
def scfi_arima_180days(df):
    scfi_Train_6m                = df[0:447]
    scfi_Test_6m                = df[447:627]
    scfi_Train_6m                = np.log(scfi_Train_6m)
    scfi_Test_6m                = np.log(scfi_Test_6m)

    scfi_Train_6m = scfi_Train_6m.reset_index()
    y_scfi_6m = scfi_Train_6m['value']
    x_scfi_6m = range(0, 447)
    scfi_Train_6m.columns.values
    scfi_lm_6m = smf.ols('y_scfi_6m ~ x_scfi_6m', data = scfi_Train_6m)
    scfi_lm_6m = scfi_lm_6m.fit()
    scfi_lm_6m.summary()
    
    scfi_stationary_6m = y_scfi_6m - scfi_lm_6m.fittedvalues
    #plot_pacf(gemval_stationary_6m)# AR1 ok, higher numbers problematics

    scfi_ar_6m = ARIMA(y_scfi_6m, order = (1, 0, 0))
    scfi_ar_6m = scfi_ar_6m.fit()



   

    scfi_ar_pred_6m2 = scfi_ar_6m.predict(start = 447, end = 627)

    
    scfi_ar_2_6m2 = SARIMAX(y_scfi_6m, order = (2, 0, 0))
    scfi_ar_2_6m2 = scfi_ar_2_6m2.fit()
    y_pred_scfi_6m2 = scfi_ar_2_6m2.get_forecast(len(scfi_Test_6m.index)+1)
    y_pred_df_scfi_6m2 = y_pred_scfi_6m2.conf_int(alpha = 0.05) 
    y_pred_df_scfi_6m2["Predictions"] = scfi_ar_2_6m2.predict(start = y_pred_df_scfi_6m2.index[0], end = y_pred_df_scfi_6m2.index[-1])
        


    scfi_ar_pred6_m2 = pd.DataFrame(scfi_ar_pred_6m2)
    scfi_Test_6m['pred_1'] = scfi_ar_pred_6m2.values[1:181]
    rmspe__scfi_1_6m2 = np.sqrt(np.sum(scfi_Test_6m.iloc[:, 0].subtract(scfi_Test_6m.iloc[:, 1])**2)/180)

    y_pred_df_scfi_6m2.columns.values
    pred_2_6m2 = pd.DataFrame(y_pred_df_scfi_6m2['Predictions'])
    scfi_Test_6m['pred_2_6m'] = pred_2_6m2.values[1:181]
    rmspe_scfi_2_6m2 = np.sqrt(np.sum(scfi_Test_6m.iloc[:, 0].subtract(scfi_Test_6m.iloc[:, 2])**2)/180)
        
    return y_scfi_6m,y_pred_df_scfi_6m2,scfi_ar_6m.summary()
    
 
    
def scfi_LSTM_7days(df):
    scfi_train_w    = df[0:620]
    #Creating Test set
    scfi_test_w     = df[620:627]
    scfi_train_w   = np.log(scfi_train_w)
    scfi_test_w     =np.log(scfi_test_w) 
    y_test=scfi_test_w['value'].values

    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler()
    lstm_scfi_w =  scaler.fit_transform(scfi_train_w)
    X_train = []
    y_train = []
    for i in range(30, len(scfi_train_w)-7):
        X_train.append(lstm_scfi_w[i-7:i, 0])
        y_train.append(lstm_scfi_w[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
     

    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))  
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    model.fit(X_train, y_train, epochs = 100, batch_size = 32)


    dataset_train_scfi_w = scfi_train_w.iloc[:613]
    dataset_test_scfi_w = scfi_train_w.iloc[613:]
    dataset_total_scfi_w = pd.concat((dataset_train_scfi_w, dataset_test_scfi_w), axis = 0)
    inputs_scfi_w = dataset_total_scfi_w[len(dataset_total_scfi_w) - len(dataset_test_scfi_w) - 7:].values
    inputs_scfi_w = inputs_scfi_w.reshape(-1,1)
    inputs_scfi_w = scaler.transform(inputs_scfi_w)
    X_test_scfi_w = []
    for i in range(7, 14):
        
        X_test_scfi_w.append(inputs_scfi_w[i-7:i, 0])
    X_test_scfi_w = np.array(X_test_scfi_w)
    X_test_scfi_w = np.reshape(X_test_scfi_w, (X_test_scfi_w.shape[0], X_test_scfi_w.shape[1], 1))
   
    pred_scfi_w = model.predict(X_test_scfi_w)
    pred_scfi_w = scaler.inverse_transform(pred_scfi_w)
    
    mae = mean_absolute_error(scfi_test_w['value'], pred_scfi_w)

    rmspe_scfi_lstm_w = np.sqrt(mean_squared_error(scfi_test_w['value'], pred_scfi_w))
    e= ['mae', 'rmspe_scfi_lstm_w']
    eval = pd.DataFrame([mae, rmspe_scfi_lstm_w], index=e, columns=['Score'])
   
    return y_test,pred_scfi_w,eval    
    
def scfi_LSTM_30days(df):
    scfi_train_m    = df[0:597]
    #Creating Test set
    scfi_test_m     = df[597:627]
    scfi_train_m   = np.log(scfi_train_m)
    scfi_test_m     =np.log(scfi_test_m) 
    y_test=scfi_test_m['value'].values
    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler()
    lstm_scfi_m =  scaler.fit_transform(scfi_train_m)
    X_train = []
    y_train = []
    for i in range(30, len(scfi_train_m)-30):
        X_train.append(lstm_scfi_m[i-30:i, 0])
        y_train.append(lstm_scfi_m[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))  
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    model.fit(X_train, y_train, epochs = 100, batch_size = 32)
    dataset_train_scfi_m = scfi_train_m.iloc[:567]
    dataset_test_scfi_m = scfi_train_m.iloc[567:]
    y_test1=dataset_test_scfi_m['value'].values
    dataset_total_scfi_w = pd.concat((dataset_train_scfi_m, dataset_test_scfi_m), axis = 0)
    inputs_scfi_m = dataset_total_scfi_w[len(dataset_total_scfi_w) - len(dataset_test_scfi_m) - 30:].values
    inputs_scfi_m = inputs_scfi_m.reshape(-1,1)
    inputs_scfi_m = scaler.transform(inputs_scfi_m)
    X_test_scfi_m = []
    for i in range(30, 60):
    
        X_test_scfi_m.append(inputs_scfi_m[i-30:i, 0])
    X_test_scfi_m = np.array(X_test_scfi_m)
    X_test_scfi_m = np.reshape(X_test_scfi_m, (X_test_scfi_m.shape[0], X_test_scfi_m.shape[1], 1))
    pred_scfi_m = model.predict(X_test_scfi_m)
    pred_scfi_m = scaler.inverse_transform(pred_scfi_m)
    mae = mean_absolute_error(scfi_test_m['value'], pred_scfi_m)
    
    rmspe_scfi_lstm_m = np.sqrt(mean_squared_error(scfi_test_m['value'], pred_scfi_m))
    
    e= ['mae', 'rmspe_scfi_lstm_m']
    eval = pd.DataFrame([mae, rmspe_scfi_lstm_m], index=e, columns=['Score'])
   
    return y_test,pred_scfi_m,eval
    
def scfi_LSTM_180days(df):    
    scfi_train_6m    = df[0:447]
    #Creating Test set
    scfi_test_6m     = df[447:627]
    scfi_train_6m    = np.log(scfi_train_6m) 
    scfi_test_6m    = np.log( scfi_test_6m)
    y_test=scfi_test_6m['value'].values    
    
    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler()
    lstm_scfi_6m =  scaler.fit_transform(scfi_train_6m)
    X_train = []
    y_train = []
    for i in range(180, len(scfi_train_6m)-180):
        X_train.append(lstm_scfi_6m[i-180:i, 0])
        y_train.append(lstm_scfi_6m[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))  
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    model.fit(X_train, y_train, epochs = 100, batch_size = 32)
    dataset_train_scfi_6m = scfi_train_6m.iloc[:267]
    dataset_test_scfi_6m = scfi_train_6m.iloc[267:]
    y_test1=dataset_test_scfi_6m['value'].values
    dataset_total_scfi_6m = pd.concat((dataset_train_scfi_6m, dataset_test_scfi_6m), axis = 0)
    inputs_scfi_6m = dataset_total_scfi_6m[len(dataset_total_scfi_6m) - len(dataset_test_scfi_6m) - 180:].values
    inputs_scfi_6m = inputs_scfi_6m.reshape(-1,1)
    inputs_scfi_6m = scaler.transform(inputs_scfi_6m)
    X_test_scfi_6m = []
    for i in range(180, 360):
        X_test_scfi_6m.append(inputs_scfi_6m[i-180:i, 0])
    X_test_scfi_6m = np.array(X_test_scfi_6m)
    X_test_scfi_6m = np.reshape(X_test_scfi_6m, (X_test_scfi_6m.shape[0], X_test_scfi_6m.shape[1], 1))
    pred_scfi_6m = model.predict(X_test_scfi_6m)
    pred_scfi_6m = scaler.inverse_transform(pred_scfi_6m)
    mae = mean_absolute_error(scfi_test_6m['value'], pred_scfi_6m)
    rmspe_scfi_lstm_6m = np.sqrt(mean_squared_error(scfi_test_6m['value'], pred_scfi_6m))
    
    e= ['mae', 'rmspe_scfi_lstm_6m']
    eval = pd.DataFrame([mae, rmspe_scfi_lstm_6m], index=e, columns=['Score'])
   
    return y_test,pred_scfi_6m,eval
def scfi_EXPO_7days(df):
    scfi_train_w    = df[0:620]
    #Creating Test set
    scfi_test_w     = df[620:627]
    scfi_train_w    = np.log(scfi_train_w)
    #Creating Test set
    scfi_train_w    = np.log(scfi_train_w)
    scfi_test_w     = np.log(scfi_test_w)
    y_test=scfi_test_w['value'].values
    scfi_model_w = ExponentialSmoothing(scfi_train_w, trend='add', seasonal=None)
    scfi_model2_w = ExponentialSmoothing(scfi_train_w, trend='add', seasonal=None, damped=True)
    scfi_fit1_w = scfi_model_w.fit()
    scfi_fit2_w = scfi_model2_w.fit()
    scfi_pred1_w = scfi_fit1_w.forecast(7)
    scfi_pred2_w = scfi_fit2_w.forecast(7)
    scfi_fit1_w.summary()
    scfi_test_w['pred_1']  =  scfi_pred1_w.values
    rmspe_scfi_Exp1_w = np.sqrt(np.sum(scfi_test_w.iloc[:, 0].subtract(scfi_test_w.iloc[:, 1])**2)/7)
    scfi_test_w['pred_2']  =  scfi_pred2_w.values
    
    mae = mean_absolute_error(scfi_test_w['value'], scfi_pred1_w)
    e= ['mae', 'rmspe_scfi_Exp1_w']
    eVal = pd.DataFrame([mae, rmspe_scfi_Exp1_w], index=e, columns=['Score'])
    rmspe_scfi_Exp2_w = np.sqrt(np.sum(scfi_test_w.iloc[:, 1].subtract(scfi_test_w.iloc[:, 2])**2)/7)
    return scfi_test_w.index,scfi_pred1_w,scfi_train_w,eVal
def scfi_EXPO_30days(df):
    scfi_train_m    = df[0:597]
    #Creating Test set
    scfi_test_m     = df[597:627]
    scfi_train_m    = np.log(scfi_train_m)
    #Creating Test set
    scfi_test_m     = np.log(scfi_test_m)
    y_test=scfi_test_m['value'].values
    scfi_model_m = ExponentialSmoothing(scfi_train_m, trend='add', seasonal=None)
    scfi_model2_m = ExponentialSmoothing(scfi_train_m, trend='add', seasonal=None, damped=True)

    scfi_fit1_m = scfi_model_m.fit()
    scfi_fit2_m = scfi_model2_m.fit()
    scfi_pred1_m = scfi_fit1_m.forecast(30)
    scfi_pred2_m = scfi_fit2_m.forecast(30)
    
    return scfi_test_m.index,scfi_pred1_m,scfi_train_m
def scfi_EXPO_180days(df):
    scfi_train_6m    = df[0:621]
    #Creating Test set
    scfi_test_6m     = df[621:627]
    scfi_train_6m    = np.log(scfi_train_6m)
    #Creating Test set
    scfi_test_6m     = np.log(scfi_test_6m)  
    y_test=scfi_test_6m['value'].values
    scfi_model_6m = ExponentialSmoothing(scfi_train_6m, trend='add', seasonal=None)
    scfi_model2_6m = ExponentialSmoothing(scfi_train_6m, trend='add', seasonal=None, damped=True)
    scfi_fit1_6m = scfi_model_6m.fit()
    scfi_fit2_6m = scfi_model2_6m.fit()
    scfi_pred1_6m = scfi_fit1_6m.forecast(6)
    scfi_pred2_6m = scfi_fit2_6m.forecast(6)
    scfi_test_6m['pred_1']  =  scfi_pred1_6m.values
    mae = mean_absolute_error(scfi_test_6m['value'], scfi_pred1_6m)
    rmspe_scfi_Exp1_m = np.sqrt(np.sum(scfi_test_6m.iloc[:, 0].subtract(scfi_test_6m.iloc[:, 1])**2)/29)
    e= ['mae', 'rmspe_scfi_Exp1_m']
    model_performance = pd.DataFrame([mae, rmspe_scfi_Exp1_m], index=e, columns=['Score'])
    
    
    return scfi_test_6m.index,scfi_pred1_6m,scfi_train_6m,model_performance
    
def gemval(df):
    gemval_train    = df[0:188]
    #Creating Test set
    gemval_test     = df[188:194]
      
    
    gemval_train_log    = np.log(gemval_train)
    #Creating Test set
    gemval_test_log     = np.log(gemval_test)
    return gemval,gemval_train,gemval_test,gemval_train_log,gemval_test_log
    
def gemval_arima_6m(train_log,gemval_test_6m):
    gemval_train_6m =train_log.reset_index()
    y_gemval_6m = gemval_train_6m['value']
    x_gemval_6m = range(0, 188)
    gemval_train_6m.columns.values
    gemval_6m = smf.ols('y_gemval_6m ~ x_gemval_6m', data =gemval_train_6m)
    gemval_6m=  gemval_6m.fit()
    gemval_6m.summary()
    #plt.plot(gemval_6m.fittedvalues)
    #plt.plot(y_gemval_6m)

    gemval_stationary_6m = y_gemval_6m - gemval_6m.fittedvalues
    #plt.plot(gemval_stationary_6m)
    plot_acf(gemval_stationary_6m)
    plot_pacf(gemval_stationary_6m) #AR1 should be good, although number 14 is a bit worring.
    #AR1 should be good, although number 14 is a bit worring.
    gemval_ar1_6m = ARIMA(y_gemval_6m, order = (1, 0, 0))
    gemval_ar1_6m = gemval_ar1_6m.fit()
    #fig = plt.figure() 
    #ax1 = fig.add_subplot(3, 1, 1) # number of row and column + position
    #ax2 = fig.add_subplot(3, 1, 2)
    #ax3 = fig.add_subplot(3, 1, 3)
    #ax1.plot(gemval_ar1_w.resid)
    #plot_acf(gemval_ar1_w.resid, ax = ax2)
    #plot_pacf(gemval_ar1_w.resid, ax = ax3)
    #fig.tight_layout()
    #plt.show() #all the short term correlation and random variation seem to be removed

    gemval_ar1_pred_6m = gemval_ar1_6m.predict(start = 188, end = 194)
    plt.plot(y_gemval_6m)
    plt.plot(gemval_ar1_pred_6m)
    # make the predictions for 1
    # plt.plot(y_gemval_6m)
    #plt.plot(gemval_ar1_pred_6m)

    gemval_ar1_2_6m = SARIMAX(y_gemval_6m, order = (1, 0, 0))
    gemval_ar1_2_6m = gemval_ar1_2_6m.fit()
    gemval_ar1_2_6m.summary() 


    y_pred_gemval_6m = gemval_ar1_2_6m.get_forecast(len(gemval_test_6m.index)+1)
    y_pred_df_gemval_6m = y_pred_gemval_6m.conf_int(alpha = 0.05) 
    y_pred_df_gemval_6m["Predictions"] = gemval_ar1_2_6m.predict(start = y_pred_df_gemval_6m.index[0], end = y_pred_df_gemval_6m.index[-1])
    #conf_df = pd.concat([test['MI'],predictions_int.predicted_mean, predictions_int.conf_int()], axis = 1)
    y_pred_gemval_6m.conf_int()
    #conf_df.head()
    #fig = plt.figure(figsize = (16,8))
    #ax1 = fig.add_subplot(1, 1, 1)
    #plt.plot(y_gemval_w)
    #plt.plot(y_pred_df_gemval_w["Predictions"],label='predicted')
    y_pred_df_gemval_6m.head()
    #plt.plot(y_pred_df_gemval_6m['lower value'], linestyle = '--', color = 'red', linewidth = 0.5,label='lower ci')
    #plt.plot(y_pred_df_gemval_6m['upper value'], linestyle = '--', color = 'red', linewidth = 0.5,label='upper ci')
    #plt.fill_between(y_pred_df_gemval_6m["Predictions"].index.values,y_pred_df_gemval_w['upper value'], color = 'grey', alpha = 0.2)
    #plt.legend(loc = 'lower left', fontsize = 12)
    #plt.show()
    gemval_ar_pred_6m = pd.DataFrame(gemval_ar1_pred_6m)
    gemval_test_6m['pred_1'] = gemval_ar_pred_6m.values[1:7]
    rmspe1_gemval_1_6m = np.sqrt(np.sum(gemval_test_6m.iloc[:, 0].subtract(gemval_test_6m.iloc[:, 1])**2)/6)
    
    gemval_ar1_6m2 = ARIMA(y_gemval_6m, order = (1, 0, 1))
    gemval_ar1_6m2 = gemval_ar1_6m2.fit()
    gemval_test_6m['pred_1'] = gemval_ar_pred_6m.values[1:7]
    rmspe1_gemval_1_6m = np.sqrt(np.sum(gemval_test_6m.iloc[:, 0].subtract(gemval_test_6m.iloc[:, 1])**2)/6)
    y_pred_df_gemval_6m.columns.values
    pred_2_6m = pd.DataFrame(y_pred_df_gemval_6m['Predictions'])
    gemval_test_6m['pred_2_w'] = pred_2_6m.values[1:7]
    rmspe1_gemval_2_6m = np.sqrt(np.sum(gemval_test_6m.iloc[:, 0].subtract(gemval_test_6m.iloc[:, 2])**2)/6)
    
    #ARIMA model (1 0 1)
    gemval_ar1_6m2 = ARIMA(y_gemval_6m, order = (1, 0, 1))
    gemval_ar1_6m2 = gemval_ar1_6m2.fit()
                                                                        
   
    gemval_ar1_pred_6m2 = gemval_ar1_6m2.predict(start = 188, end = 194)
    # plt.plot(y_gemval_6m)
    # plt.plot(gemval_ar1_pred_6m2)

    gemval_ar1_2_6m2 = SARIMAX(y_gemval_6m, order = (1, 0, 1)) 
    gemval_ar1_2_6m2 = gemval_ar1_2_6m2.fit()
    y_pred_gemval_6m2 = gemval_ar1_2_6m2.get_forecast(len(gemval_test_6m.index)+1)
    y_pred_df_gemval_6m2 = y_pred_gemval_6m2.conf_int(alpha = 0.05) 
    y_pred_df_gemval_6m2["Predictions"] = gemval_ar1_2_6m2.predict(start = y_pred_df_gemval_6m2.index[0], end = y_pred_df_gemval_6m2.index[-1])

    gemval_ar_pred_6m2 = pd.DataFrame(gemval_ar1_pred_6m2)
    gemval_test_6m['pred_1'] = gemval_ar_pred_6m2.values[1:7]
    rmspe2_gemval_1_6m = np.sqrt(np.sum(gemval_test_6m.iloc[:, 0].subtract(gemval_test_6m.iloc[:, 1])**2)/6)
    y_pred_df_gemval_6m2.columns.values
    pred_2_6m2 = pd.DataFrame(y_pred_df_gemval_6m2['Predictions'])
    gemval_test_6m['pred_2_w'] = pred_2_6m2.values[1:7]
    rmspe2_gemval_2_6m = np.sqrt(np.sum(gemval_test_6m.iloc[:, 0].subtract(gemval_test_6m.iloc[:, 2])**2)/6)
    return y_gemval_6m,y_pred_df_gemval_6m2,gemval_ar1_2_6m.summary()
    
def gemval_arima_1year(df):
    gemval_Train_1y    = df[0:182]
    gemval_Test_1y     = df[182:194]
    gemval_Train_1y    = np.log(gemval_Train_1y)
    gemval_Test_1y    = np.log(gemval_Test_1y)
    
    gemval_Train_1y = gemval_Train_1y.reset_index()
    y_gemval_1y = gemval_Train_1y['value']
    x_gemval_1y = range(0, 182)
    gemval_lm_1y = smf.ols('y_gemval_1y ~ x_gemval_1y', data = gemval_Train_1y)
    gemval_lm_1y = gemval_lm_1y.fit()
    gemval_lm_1y.summary()
    #plt.plot(gemval_lm_1y.fittedvalues)
    #plt.plot(y_gemval_1y)
    

    gemval_stationary_1y = y_gemval_1y - gemval_lm_1y.fittedvalues
   

    gemval_ar_1y = ARIMA(y_gemval_1y, order = (1, 0, 0))
    gemval_ar_1y = gemval_ar_1y.fit()


    

    gemval_ar_pred_1y = gemval_ar_1y.predict(start = 182, end = 194)


    
    gemval_ar_2_1y = SARIMAX(y_gemval_1y, order = (1, 0, 0))

    gemval_ar_2_1y = gemval_ar_2_1y.fit()
    y_pred_gemval_1y = gemval_ar_2_1y.get_forecast(len(gemval_Test_1y.index)+1)
    y_pred_df_gemval_1y = y_pred_gemval_1y.conf_int(alpha = 0.05) 
    y_pred_df_gemval_1y["Predictions"] = gemval_ar_2_1y.predict(start = y_pred_df_gemval_1y.index[0], end = y_pred_df_gemval_1y.index[-1])
        
    gemval_ar_pred_1y = pd.DataFrame(gemval_ar_pred_1y)
    gemval_Test_1y['pred_1'] = gemval_ar_pred_1y.values[1:13]
    rmspe_gemval_1_1y = np.sqrt(np.sum(gemval_Test_1y.iloc[:, 0].subtract(gemval_Test_1y.iloc[:, 1])**2)/12)

    pred_2_1y = pd.DataFrame(y_pred_df_gemval_1y['Predictions'])
    gemval_Test_1y['pred_2_1y'] = pred_2_1y.values[1:13]
    rmspe__gemval_2_1y = np.sqrt(np.sum(gemval_Test_1y.iloc[:, 0].subtract(gemval_Test_1y.iloc[:, 2])**2)/12)


    #Arima(1 0 1)

    gemval_ar1_1y2 = ARIMA(y_gemval_1y, order = (1, 0, 1))
    gemval_ar1_1y2 = gemval_ar1_1y2.fit()

    
    gemval_ar1_pred_1y2 = gemval_ar1_1y2.predict(start = 182, end = 194)
    
    gemval_ar1_2_1y2 = SARIMAX(y_gemval_1y, order = (1, 0, 1)) 
    gemval_ar1_2_1y2 = gemval_ar1_2_1y2.fit()
    y_pred_gemval_1y2 = gemval_ar1_2_1y2.get_forecast(len(gemval_Test_1y.index)+1)
    y_pred_df_gemval_1y2 = y_pred_gemval_1y2.conf_int(alpha = 0.05) 
    y_pred_df_gemval_1y2["Predictions"] = gemval_ar1_2_1y2.predict(start = y_pred_df_gemval_1y2.index[0], end = y_pred_df_gemval_1y2.index[-1])
    '''fig = plt.figure(figsize = (16,8))
    ax1 = fig.add_subplot(1, 1, 1)
    plt.plot(y_gemval_m)
    plt.plot(y_pred_df_gemval_m2["Predictions"],color = 'red',label='predicted')
    y_pred_df_gemval_m2.head()
    plt.plot(y_pred_df_gemval_m2['lower value'], linestyle = '--', color = 'red', linewidth = 0.5, label='lower ci')
    plt.plot(y_pred_df_gemval_m2['upper value'], linestyle = '--', color = 'red', linewidth = 0.5, label='upper ci')
    plt.fill_between(y_pred_df_gemval_m2["Predictions"].index.values,
                     y_pred_df_gemval_m2['lower value'], 
                     y_pred_df_gemval_m2['upper value'], 
                     color = 'grey', alpha = 0.2)
    plt.legend(loc = 'best')
    plt.show()'''

    gemval_ar_pred_1y2 = pd.DataFrame(gemval_ar1_pred_1y2)
    gemval_Test_1y['pred_1'] = gemval_ar_pred_1y2.values[1:13]
    rmspe2_gemval_1_1y = np.sqrt(np.sum(gemval_Test_1y.iloc[:, 0].subtract(gemval_Test_1y.iloc[:, 1])**2)/12)

    y_pred_df_gemval_1y2.columns.values
    pred_2_1y2 = pd.DataFrame(y_pred_df_gemval_1y2['Predictions'])
    gemval_Test_1y['pred_2_m'] = pred_2_1y2.values[1:13]
    rmspe2_gemval_2_1y = np.sqrt(np.sum(gemval_Test_1y.iloc[:, 0].subtract(gemval_Test_1y.iloc[:, 2])**2)/12)


    return y_gemval_1y,y_pred_df_gemval_1y2,gemval_ar_1y.summary()
    
def gemval_arima_2years(df):
    gemval_Train_2y    = df[0:170]
    gemval_Test_2y    = df[170:194]
    gemval_Train_2y    = np.log(gemval_Train_2y)
    gemval_Test_2y    = np.log(gemval_Test_2y)
    

    gemval_Train_2y = gemval_Train_2y.reset_index()
    y_gemval_2y = gemval_Train_2y['value']
    x_gemval_2y = range(0, 170)
    gemval_Train_2y.columns.values
    gemval_lm_2y = smf.ols('y_gemval_2y ~ x_gemval_2y', data = gemval_Train_2y)
    gemval_lm_2y = gemval_lm_2y.fit()
    gemval_lm_2y.summary()
    
    gemval_stationary_2y = y_gemval_2y - gemval_lm_2y.fittedvalues
    #plot_pacf(gemval_stationary_6m)# AR1 ok, higher numbers problematics

    gemval_ar_2y = ARIMA(y_gemval_2y, order = (1, 0, 0))
    gemval_ar_2y = gemval_ar_2y.fit()

   

    gemval_ar_pred_2y2 = gemval_ar_2y.predict(start = 170, end = 194)

    
    gemval_ar_2_2y2 = SARIMAX(y_gemval_2y, order = (1, 0, 0))
    gemval_ar_2_2y2 = gemval_ar_2_2y2.fit()
    y_pred_gemval_2y2 = gemval_ar_2_2y2.get_forecast(len(gemval_Test_2y.index)+1)
    y_pred_df_gemval_2y2 = y_pred_gemval_2y2.conf_int(alpha = 0.05) 
    y_pred_df_gemval_2y2["Predictions"] = gemval_ar_2_2y2.predict(start = y_pred_df_gemval_2y2.index[0], end = y_pred_df_gemval_2y2.index[-1])



    gemval_ar_pred_2y2 = pd.DataFrame(gemval_ar_pred_2y2)
    gemval_Test_2y['pred_1'] = gemval_ar_pred_2y2.values[1:25]
    rmspe__gemval_1_2y2 = np.sqrt(np.sum(gemval_Test_2y.iloc[:, 0].subtract(gemval_Test_2y.iloc[:, 1])**2)/24)

    y_pred_df_gemval_2y2.columns.values
    pred_2_2y2 = pd.DataFrame(y_pred_df_gemval_2y2['Predictions'])
    gemval_Test_2y['pred_2_2y'] = pred_2_2y2.values[1:25]
    rmspe_gemval_2_2y2 = np.sqrt(np.sum(gemval_Test_2y.iloc[:, 0].subtract(gemval_Test_2y.iloc[:, 2])**2)/24)


    
    return y_gemval_2y,y_pred_df_gemval_2y2,gemval_ar_2y.summary()
    

def LSTM_1year(df):
    gemval_Train_1y    = df[0:182]
    gemval_Test_1y     = df[182:194]
    gemval_Train_1y    = np.log(gemval_Train_1y)
    gemval_Test_1y    = np.log(gemval_Test_1y)
    y_test=gemval_Test_1y['value'].values

    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler()
    lstm_gemval_1y = scaler.fit_transform(gemval_Train_1y)
    X_train = []
    y_train = []
    for i in range(12, len(gemval_Train_1y)-12):
        X_train.append(lstm_gemval_1y[i-12:i, 0])
        y_train.append(lstm_gemval_1y[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    X_train.shape

    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))  
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    model.fit(X_train, y_train, epochs = 100, batch_size = 32)


    dataset_train_gemval_1y = gemval_Train_1y.iloc[:170]
    dataset_test_gemval_1y = gemval_Train_1y.iloc[170:]
    dataset_total_gemval_1y = pd.concat((dataset_train_gemval_1y, dataset_test_gemval_1y), axis = 0)
    inputs_gemval_1y = dataset_total_gemval_1y[len(dataset_total_gemval_1y) - len(dataset_test_gemval_1y) - 12:].values
    inputs_gemval_1y = inputs_gemval_1y.reshape(-1,1)
    inputs_gemval_1y = scaler.transform(inputs_gemval_1y)
    X_test_gemval_1y = []
    for i in range(12, 24):
        
        X_test_gemval_1y.append(inputs_gemval_1y[i-12:i, 0])
    X_test_gemval_1y = np.array(X_test_gemval_1y)
    X_test_gemval_1y = np.reshape(X_test_gemval_1y, (X_test_gemval_1y.shape[0], X_test_gemval_1y.shape[1], 1))
    print(X_test_gemval_1y.shape)

    pred_gemval_1y = model.predict(X_test_gemval_1y)
    pred_gemval_1y = scaler.inverse_transform(pred_gemval_1y)

    
    mae = mean_absolute_error(gemval_Test_1y['value'], pred_gemval_1y)

    rmspe_gemval_lstm_1y = np.sqrt(mean_squared_error(gemval_Test_1y['value'], pred_gemval_1y))
    e= ['mae', 'rmspe_gemval_lstm_1y']
    eval = pd.DataFrame([mae, rmspe_gemval_lstm_1y], index=e, columns=['Score'])
   
    return y_test,pred_gemval_1y,eval
    
def LSTM_2years(df):
    gemval_Train_2y    = df[0:170]
    gemval_Test_2y    = df[170:194]
    gemval_Train_2y    = np.log(gemval_Train_2y)
    gemval_Test_2y    = np.log(gemval_Test_2y)
    y_test=gemval_Test_2y['value'].values

    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler()
    lstm_gemval_2y = scaler.fit_transform(gemval_Train_2y)
    X_train = []
    y_train = []
    for i in range(24, len(gemval_Train_2y)-24):
         X_train.append(lstm_gemval_2y[i-24:i, 0])
         y_train.append(lstm_gemval_2y[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    X_train.shape

    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))  
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    model.fit(X_train, y_train, epochs = 100, batch_size = 32)


    dataset_train_gemval_2y = gemval_Train_2y.iloc[:146]
    dataset_test_gemval_2y = gemval_Train_2y.iloc[146:]
    dataset_total_gemval_2y = pd.concat((dataset_train_gemval_2y, dataset_test_gemval_2y), axis = 0)
    inputs_gemval_2y = dataset_total_gemval_2y[len(dataset_total_gemval_2y) - len(dataset_test_gemval_2y) - 24:].values
    inputs_gemval_2y = inputs_gemval_2y.reshape(-1,1)
    inputs_gemval_2y = scaler.transform(inputs_gemval_2y)
    X_test_gemval_2y = []
    for i in range(24, 48):
        
        X_test_gemval_2y.append(inputs_gemval_2y[i-24:i, 0])
    X_test_gemval_2y = np.array(X_test_gemval_2y)
    X_test_gemval_2y = np.reshape(X_test_gemval_2y, (X_test_gemval_2y.shape[0], X_test_gemval_2y.shape[1], 1))
    print(X_test_gemval_2y.shape)
    pred_gemval_2y = model.predict(X_test_gemval_2y)
    pred_gemval_2y = scaler.inverse_transform(pred_gemval_2y)

    
    mae = mean_absolute_error(gemval_Test_2y['value'], pred_gemval_2y)

    rmspe_gemval_lstm_2y = np.sqrt(mean_squared_error(gemval_Test_2y['value'], pred_gemval_2y))
    e= ['mae', 'rmspe_gemval_lstm_2y']
    eval = pd.DataFrame([mae, rmspe_gemval_lstm_2y], index=e, columns=['Score'])
   
    return y_test,pred_gemval_2y,eval
    
    
def LSTM_6months(df):    
    gemval_train_6m    = df[0:188]
    #Creating Test set
    gemval_test_6m     = df[188:194]
    gemval_train_6m    = np.log(gemval_train_6m)
    #Creating Test set
    gemval_test_6m     = np.log(gemval_test_6m)
    y_test=gemval_test_6m['value'].values
    
    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler()
    lstm_gemval_6m = scaler.fit_transform(gemval_train_6m)
    X_train = []
    y_train = []
    for i in range(6, len(gemval_train_6m)-6):
        X_train.append(lstm_gemval_6m[i-6:i, 0])
        y_train.append(lstm_gemval_6m[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))  
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))         
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    model.fit(X_train, y_train, epochs = 100, batch_size = 32)

    dataset_train_gemval_6m = gemval_train_6m.iloc[:182]
    dataset_test_gemval_6m = gemval_train_6m.iloc[182:]
    dataset_total_gemval_6m = pd.concat((dataset_train_gemval_6m, dataset_test_gemval_6m), axis = 0)
    inputs_gemval_6m = dataset_total_gemval_6m[len(dataset_total_gemval_6m) - len(dataset_test_gemval_6m) - 6:].values
    inputs_gemval_6m = inputs_gemval_6m.reshape(-1,1)
    inputs_gemval_6m = scaler.transform(inputs_gemval_6m)
    X_test_gemval_6m = []
    for i in range(6, 12):
        X_test_gemval_6m.append(inputs_gemval_6m[i-6:i, 0])
    X_test_gemval_6m = np.array(X_test_gemval_6m)
    X_test_gemval_6m = np.reshape(X_test_gemval_6m, (X_test_gemval_6m.shape[0], X_test_gemval_6m.shape[1], 1))
    print(X_test_gemval_6m.shape)

    pred_gemval_6m = model.predict(X_test_gemval_6m)
    pred_gemval_6m = scaler.inverse_transform(pred_gemval_6m)
    mae = mean_absolute_error(gemval_test_6m['value'],  pred_gemval_6m)
    
    rmspe_gemval_lstm_6m = np.sqrt(mean_squared_error(gemval_test_6m['value'], pred_gemval_6m))
    e= ['mae', 'rmspe_gemval_lstm_6m']
    eval = pd.DataFrame([mae, rmspe_gemval_lstm_6m], index=e, columns=['Score'])
   
    
   
    return y_test,pred_gemval_6m,eval

def EXPO_6months(df): 
    gemval_train_6m    = df[0:188]
    #Creating Test set
    gemval_test_6m     = df[188:194]
    gemval_train_6m   = np.log(gemval_train_6m)
    gemval_test_6m     =np.log(gemval_test_6m) 
    y_test=gemval_test_6m['value'].values
    gemval_model_6m = ExponentialSmoothing(gemval_train_6m, trend='add', seasonal=None)
    gemval_model2_6m = ExponentialSmoothing(gemval_train_6m, trend='add', seasonal=None, damped=True)

    gemval_fit1_6m = gemval_model_6m.fit()
    gemval_fit2_6m = gemval_model2_6m.fit()
    gemval_pred1_6m = gemval_fit1_6m.forecast(6)
    gemval_pred2_6m = gemval_fit2_6m.forecast(6)
 
    gemval_test_6m['pred_1']  =  gemval_pred1_6m.values
    mae = mean_absolute_error(gemval_test_6m ['value'], gemval_pred1_6m)
    rmspe_gemval_Exp1_6m = np.sqrt(np.sum(gemval_test_6m.iloc[:, 0].subtract(gemval_test_6m.iloc[:, 1])**2)/6)
    gemval_test_6m['pred_2']  =  gemval_pred2_6m.values
    
   
    e= ['mae', 'rmspe_gemval_Exp1_6m']
    model_performance = pd.DataFrame([mae, rmspe_gemval_Exp1_6m], index=e, columns=['Score'])
    
    
    return gemval_test_6m.index,gemval_pred1_6m,gemval_train_6m,model_performance
    
def EXPO_1y(df):    
    gemval_Train_1y    = df[0:182]
    gemval_Test_1y     = df[182:194]
    gemval_Train_1y    = np.log(gemval_Train_1y)
    gemval_Test_1y    = np.log(gemval_Test_1y)
    y_test=gemval_Test_1y['value'].values
    gemval_model_1y = ExponentialSmoothing(gemval_Train_1y, trend='add', seasonal=None)
    gemval_model2_1y = ExponentialSmoothing(gemval_Train_1y, trend='add', seasonal=None, damped=True)

    gemval_fit1_1y = gemval_model_1y.fit()
    gemval_fit2_1y = gemval_model2_1y.fit()
    gemval_pred1_1y = gemval_fit1_1y.forecast(12)
    gemval_pred2_1y = gemval_fit2_1y.forecast(12)
 
 
    gemval_Test_1y['pred_1']  =  gemval_pred1_1y.values
    rmspe_gemval_Exp1_1y = np.sqrt(np.sum(gemval_Test_1y.iloc[:, 0].subtract(gemval_Test_1y.iloc[:, 1])**2)/12)
    mae = mean_absolute_error(gemval_Test_1y ['value'], gemval_pred1_1y)
    e= ['mae', 'rmspe_gemval_Exp1_1y']
    model_performance = pd.DataFrame([mae, rmspe_gemval_Exp1_1y], index=e, columns=['Score'])
    
    gemval_Test_1y['pred_2']  =  gemval_pred2_1y.values
    rmspe_gemval_Exp2_1y = np.sqrt(np.sum(gemval_Test_1y.iloc[:, 1].subtract(gemval_Test_1y.iloc[:, 2])**2)/12)
    return gemval_Test_1y.index,gemval_pred1_1y,gemval_Train_1y,model_performance
    
def EXPO_2y(df):   
    gemval_Train_2y    = df[0:170]
    gemval_Test_2y    = df[170:194]
    gemval_Train_2y    = np.log(gemval_Train_2y)
    gemval_Test_2y    = np.log(gemval_Test_2y)
    y_test=gemval_Test_2y['value'].values
    gemval_model_2y = ExponentialSmoothing(gemval_Train_2y, trend='add', seasonal=None)
    gemval_model2_2y = ExponentialSmoothing(gemval_Train_2y, trend='add', seasonal=None, damped=True)  
    gemval_fit1_2y = gemval_model_2y.fit()
    gemval_fit2_2y = gemval_model2_2y.fit()
    gemval_pred1_2y = gemval_fit1_2y.forecast(24)
    gemval_pred2_2y = gemval_fit2_2y.forecast(24)
    gemval_Test_2y['pred_1']  =  gemval_pred1_2y.values
    rmspe_gemval_Exp1_2y = np.sqrt(np.sum(gemval_Test_2y.iloc[:, 0].subtract(gemval_Test_2y.iloc[:, 1])**2)/24)
    mae = mean_absolute_error(gemval_Test_2y ['value'], gemval_pred1_2y)
    e= ['mae', 'rmspe_gemval_Exp1_2y']
    model_performance = pd.DataFrame([mae, rmspe_gemval_Exp1_2y], index=e, columns=['Score'])
    
    gemval_Test_2y['pred_2']  =  gemval_pred2_2y.values
    rmspe_gemval_Exp2_2y = np.sqrt(np.sum(gemval_Test_2y.iloc[:, 0].subtract(gemval_Test_2y.iloc[:, 2])**2)/24)

    return gemval_Test_2y.index,  gemval_pred1_2y,gemval_Train_2y,model_performance
        




                
            