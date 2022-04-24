import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import scfi_gemval
import time
from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.support.ui import WebDriverWait

st.markdown("<h1 style='text-align: center; color:#F08080 ;'>SKYMINERALS DASHBOARD </h1>", unsafe_allow_html=True)
Datasets=st.sidebar.selectbox("Datasets", ["scfi","gemval"])
Models=st.sidebar.selectbox("Models", ["ARIMA","LSTM","EXPO"])
Prediction_Period=st.sidebar.selectbox("Prediction_Period", ["7 Days","30 Days","180 Days","1y","2y"])
Refresh=st.sidebar.button("Refresh")
if Refresh: 
    if Datasets=="scfi":
        try: 
            url="https://en.macromicro.me/collections/3208/high-frequency-data/947/commodity-ccfi-scfi"

            #wait = WebDriverWait(driver, 30)
            driver=webdriver.Firefox(executable_path=GeckoDriverManager().install())
            driver = webdriver.Firefox(executable_path='C:/Users/Vijaya/.wdm/drivers/geckodriver/win64/v0.31.0/geckodriver-v0.31.0-win64/geckodriver.exe')
         
            wait = WebDriverWait(driver, 30)
            driver.get(url)
            
            #driver.execute_script("document.getElementById('highcharts-rh23pi0-0').scrollIntoView()")

        # wait until the chart div has been rendered before accessing the data provider
            wait.until(lambda x: x.find_element_by_class_name("highcharts-container ").is_displayed())
            time.sleep(5)
            temp=driver.execute_script("return Highcharts.charts[0].series[1].options.data")
            df = pd.DataFrame(temp).set_axis(['date', 'value'], axis=1, inplace=False)  # Convert data to DataFrame object
            
            df['date'] = pd.to_datetime(df['date'], unit='ms')  # Convert timestamp to date
            df.to_csv('freight_index.csv', index=False)
            #gemval=pd.read_csv("gemval_index.csv")
            driver.close()
        except Exception as e:
            print(e)
        st.markdown("<h3 style='text-align: Left; color:  #CA6F1E;'>Dataset-scfi</h3>", unsafe_allow_html=True)
        scfi=pd.read_csv("https://raw.githubusercontent.com/vijayapaluri/minerals-lgistics/main/freight_index.csv",parse_dates = ['date'], index_col = ['date'])
        st.write(scfi) 
        st.markdown("<h3 style='text-align: Left; color:  #CA6F1E;'>Summary Statistics</h3>", unsafe_allow_html=True) 
        st.dataframe(scfi.describe())
        st.dataframe(scfi.skew())
        st.dataframe(scfi.kurt())
        st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Before Predictions on Train values</h5>", unsafe_allow_html=True)
        fig=plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_facecolor('#EAF2F8')
        plt.plot(scfi)
        plt.xlabel("date")
        plt.ylabel("value")
        plt.legend(['actual','values'])
        st.pyplot(fig)
        dataset,train_w,test_w,train_w_log,test_w_log=scfi_gemval.scfi(scfi)
        if Models=="ARIMA":
            if Prediction_Period=="7 Days":
                y_scfi_w,y_pred_df_scfi_w,y_fit=scfi_gemval.scfi_arima_7days(train_w_log,test_w_log)
                st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 7 days</h5>", unsafe_allow_html=True)
                fig = plt.figure(figsize = (16,8))
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.set_facecolor('#EAF2F8')
                plt.title("Confidence Interval after 7 days")
                plt.plot(y_scfi_w)
                plt.plot(y_pred_df_scfi_w["Predictions"],label='predicted')
                
                plt.plot(y_pred_df_scfi_w['lower value'], linestyle = '--', color = 'red', linewidth = 0.5,label='lower ci')
                plt.plot(y_pred_df_scfi_w['upper value'], linestyle = '--', color = 'red', linewidth = 0.5,label='upper ci')
                plt.fill_between(y_pred_df_scfi_w["Predictions"].index.values,
                                 y_pred_df_scfi_w['upper value'], 
                                 color = 'grey', alpha = 0.2)
                plt.legend(loc = 'lower left', fontsize = 12)
                st.pyplot(fig)
                st.write(y_fit)
                
       
            elif Prediction_Period=="30 Days":
                y_scfi_m,y_pred_df_scfi_m,y_fit=scfi_gemval.scfi_arima_30days(scfi)
                
                st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 30 days</h5>", unsafe_allow_html=True)
                fig = plt.figure(figsize = (16,8))
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.set_facecolor('#EAF2F8')
                plt.title("Confidence Interval after 30 days")
                plt.plot(y_scfi_m)
                plt.plot(y_pred_df_scfi_m["Predictions"],label='predicted')
                plt.plot(y_pred_df_scfi_m['lower value'], linestyle = '--', color = 'red', linewidth = 0.5,label='lower ci')
                plt.plot(y_pred_df_scfi_m['upper value'], linestyle = '--', color = 'red', linewidth = 0.5,label='upper ci')
                plt.fill_between(y_pred_df_scfi_m["Predictions"].index.values,
                                 y_pred_df_scfi_m['upper value'], 
                                 color = 'grey', alpha = 0.2)
                plt.legend(loc = 'lower left', fontsize = 12)
                st.pyplot(fig)
                st.write(y_fit)
            elif Prediction_Period=="180 Days":
                     y_scfi_m,y_pred_df_scfi_m,y_fit=scfi_gemval.scfi_arima_180days(scfi)
                     
                     st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 180 days</h5>", unsafe_allow_html=True)
                     fig = plt.figure(figsize = (16,8))
                     ax1 = fig.add_subplot(1, 1, 1)
                     ax1.set_facecolor('#EAF2F8')
                     plt.title("Confidence Interval after 180 days")
                     plt.plot(y_scfi_m)
                     plt.plot(y_pred_df_scfi_m["Predictions"],label='predicted')
                     plt.plot(y_pred_df_scfi_m['lower value'], linestyle = '--', color = 'red', linewidth = 0.5,label='lower ci')
                     plt.plot(y_pred_df_scfi_m['upper value'], linestyle = '--', color = 'red', linewidth = 0.5,label='upper ci')
                     plt.fill_between(y_pred_df_scfi_m["Predictions"].index.values,
                                      y_pred_df_scfi_m['upper value'], 
                                      color = 'grey', alpha = 0.2)
                     plt.legend(loc = 'lower left', fontsize = 12)
                     st.pyplot(fig)
                     st.write(y_fit)
            elif Prediction_Period=="1y":
                     st.markdown("<h3 style='text-align: Left; color:  #CA6F1E;'>Dataset-scfi has less number of values can't predict  </h3>", unsafe_allow_html=True)
            elif Prediction_Period=="2y":
                     st.markdown("<h3 style='text-align: Left; color:  #CA6F1E;'>Dataset-scfi has less number of values can't predict  </h3>", unsafe_allow_html=True)
        elif Models=="LSTM":
            if Prediction_Period=="7 Days":
                y_test,pred_scfi_w,eval=scfi_gemval.scfi_LSTM_7days(scfi)
                st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 7 days</h5>", unsafe_allow_html=True)
                fig = plt.figure(figsize = (16,8))
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.set_facecolor('#EAF2F8')
                plt.title("Confidence Interval after 7 days")
                plt.plot(y_test)
                plt.plot(pred_scfi_w,label='predicted')
                plt.legend(loc = 'lower left', fontsize = 12)
                st.pyplot(fig)
                st.markdown("<h5 style='text-align: Left; color:  #EC7063;'>Model Performance</h5>", unsafe_allow_html=True)
                st.write(eval)
            elif Prediction_Period=="30 Days":
                y_test,pred_scfi_w,eval=scfi_gemval.scfi_LSTM_30days(scfi)
                st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 30 days</h5>", unsafe_allow_html=True)
                fig = plt.figure(figsize = (16,8))
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.set_facecolor('#EAF2F8')
                plt.title("Confidence Interval after 30 days")
                plt.plot(y_test)
                plt.plot(pred_scfi_w,label='predicted')
                plt.legend(loc = 'lower left', fontsize = 12)
                st.pyplot(fig)
                st.markdown("<h5 style='text-align: Left; color:  #EC7063;'>Model Performance</h5>", unsafe_allow_html=True)
                st.write(eval)
            elif Prediction_Period=="180 Days":
                y_test,pred_scfi_w,eval=scfi_gemval.scfi_LSTM_180days(scfi)
                st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 180 days</h5>", unsafe_allow_html=True)
                fig = plt.figure(figsize = (16,8))
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.set_facecolor('#EAF2F8')
                plt.title("Confidence Interval after 180 days")
                plt.plot(y_test)
                plt.plot(pred_scfi_w,label='predicted')
                plt.legend(loc = 'lower left', fontsize = 12)
                st.pyplot(fig)
                st.markdown("<h5 style='text-align: Left; color:  #EC7063;'>Model Performance</h5>", unsafe_allow_html=True)
                st.write(eval)
            elif Prediction_Period=="1y":
                st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Dataset-scfi has less number of values can't predict</h5>", unsafe_allow_html=True)
            elif Prediction_Period=="2y":
                st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Dataset-scfi has less number of values can't predict</h5>", unsafe_allow_html=True)
        elif Models=="EXPO":
            if Prediction_Period=="7 Days":
                scfi_test_w,scfi_pred1_w,scfi_train_w,eVal=scfi_gemval.scfi_EXPO_7days(scfi)
                st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 7 days</h5>", unsafe_allow_html=True)
                fig = plt.figure(figsize = (16,8))
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.set_facecolor('#EAF2F8')
                plt.plot(scfi_train_w, label='Train')
                plt.plot(scfi_test_w, scfi_pred1_w, label='Exponential Smoothing ')
                plt.legend(loc = 'best')
                st.pyplot(fig)
                st.markdown("<h5 style='text-align: Left; color:  #EC7063;'>Model Performance</h5>", unsafe_allow_html=True)
                st.write(eVal)
            elif Prediction_Period=="30 Days":
                scfi_test_m,scfi_pred1_m,scfi_train_m=scfi_gemval.scfi_EXPO_30days(scfi)
                st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 30 days</h5>", unsafe_allow_html=True)
                fig = plt.figure(figsize = (16,8))
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.set_facecolor('#EAF2F8')
                plt.plot(scfi_train_m, label='Train')
                plt.plot(scfi_test_m, scfi_pred1_m, label='Exponential Smoothing')
                plt.legend(loc = 'best')
                st.pyplot(fig)
               
             
            elif Prediction_Period=="180 Days":
                scfi_test_6m,scfi_pred1_6m,scfi_train_6m,model_performance =scfi_gemval.scfi_EXPO_180days(scfi)
                st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 180 days</h5>", unsafe_allow_html=True)
                fig = plt.figure(figsize = (16,8))
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.set_facecolor('#EAF2F8')
               
                plt.plot(scfi_train_6m, label='Train')
                plt.plot(scfi_test_6m, scfi_pred1_6m, label='Exponential Smoothing ')
                plt.legend(loc = 'best')
                st.pyplot(fig)
                st.markdown("<h5 style='text-align: Left; color:  #EC7063;'>Model Performance</h5>", unsafe_allow_html=True)
                st.write(model_performance)
            elif Prediction_Period=="1y":
                st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Data set has less data values so can't predict</h5>", unsafe_allow_html=True) 
            elif Prediction_Period=="2y":
                st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Data set has less data values so can't predict</h5>", unsafe_allow_html=True) 
                
    elif Datasets=="gemval":
                try: 
                    url="https://gemval.com/gva/?index=GVA&term=2"
                    #wait = WebDriverWait(driver, 30)
                    driver=webdriver.Firefox(executable_path=GeckoDriverManager().install())
                    driver = webdriver.Firefox(executable_path='C:/Users/Vijaya/.wdm/drivers/geckodriver/win64/v0.31.0/geckodriver-v0.31.0-win64/geckodriver.exe')
                 
                    wait = WebDriverWait(driver, 30)
                    driver.get(url)
                    
                    driver.execute_script("document.getElementById('gemval-aggregate-chart').scrollIntoView()")
            
                # wait until the chart div has been rendered before accessing the data provider
                    wait.until(lambda x: x.find_element_by_class_name("amcharts-chart-div").is_displayed())
                    time.sleep(5)
                    temp=driver.execute_script("return AmCharts.charts[0].dataProvider")
                    df = pd.DataFrame(temp).set_axis(['date', 'value'], axis=1, inplace=False)  # Convert data to DataFrame object
                    
                    df['date'] = pd.to_datetime(df['date'], unit='ms')  # Convert timestamp to date
                    df.to_csv('gemval_index.csv', index=False)
                    #gemval=pd.read_csv("gemval_index.csv")
                    driver.close()
                except Exception as e:
                    print(e)
                st.markdown("<h3 style='text-align: Left; color:  #CA6F1E;'>Dataset-gemval</h3>", unsafe_allow_html=True)
                gemval=pd.read_csv("https://raw.githubusercontent.com/vijayapaluri/minerals-lgistics/main/gemval_index.csv",parse_dates = ['date'], index_col = ['date'])
                #gemval=pd.read_csv("C:/Users/Vijaya/gemvalue/gemval_index.csv",parse_dates = ['date'], index_col = ['date'])
                st.write(gemval) 
                st.markdown("<h3 style='text-align: Left; color:  #CA6F1E;'>Summary Statistics</h3>", unsafe_allow_html=True) 
                st.dataframe(gemval.describe())
                st.dataframe(gemval.skew())
                st.dataframe(gemval.kurt())
                st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Before Predictions on Train values</h5>", unsafe_allow_html=True)
                fig=plt.figure(figsize=(12,6))
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.set_facecolor('#EAF2F8')
                plt.plot(gemval)
                plt.xlabel("date")
                plt.ylabel("value")
                plt.legend(['actual','values'])
                st.pyplot(fig)
                dataset,train,test,train_6m_log,test_6m_log=scfi_gemval.gemval(gemval)
                if Models=="ARIMA":   
                    if Prediction_Period=="7 Days":
                       st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Data set has weekly data values so can't predict</h5>", unsafe_allow_html=True) 
                    if Prediction_Period=="30 Days":
                       st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Data set has weekly data values so can't predict</h5>", unsafe_allow_html=True)  
                    if Prediction_Period=="180 Days":
                        y_gemval_6m,y_pred_df_gemval_6m2,gemval_ar1_2_6m=scfi_gemval.gemval_arima_6m(train_6m_log,test)
                        st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 180 days</h5>", unsafe_allow_html=True)
                        fig = plt.figure(figsize = (16,8))
                        ax1 = fig.add_subplot(1, 1, 1)
                        ax1.set_facecolor('#EAF2F8')
                        plt.title("Confidence Interval after 180 days")
                        plt.plot(y_gemval_6m)
                        plt.plot(y_pred_df_gemval_6m2["Predictions"],label='predicted')
                        
                        plt.plot(y_pred_df_gemval_6m2['lower value'], linestyle = '--', color = 'red', linewidth = 0.5,label='lower ci')
                        plt.plot(y_pred_df_gemval_6m2['upper value'], linestyle = '--', color = 'red', linewidth = 0.5,label='upper ci')
                        plt.fill_between(y_pred_df_gemval_6m2["Predictions"].index.values,
                                         y_pred_df_gemval_6m2['upper value'], 
                                         color = 'grey', alpha = 0.2)
                        plt.legend(loc = 'lower left', fontsize = 12)
                        st.pyplot(fig)
                        st.write(gemval_ar1_2_6m)
                    elif Prediction_Period=="1y":
                        y_gemval_1y,y_pred_df_gemval_1y2,gemval_ar1_2_1y=scfi_gemval.gemval_arima_1year(gemval)
                        st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 1year</h5>", unsafe_allow_html=True)
                        fig = plt.figure(figsize = (16,8))
                        ax1 = fig.add_subplot(1, 1, 1)
                        ax1.set_facecolor('#EAF2F8')
                        plt.title("Confidence Interval after 1 year")
                        plt.plot(y_gemval_1y)
                        plt.plot(y_pred_df_gemval_1y2["Predictions"],label='predicted')
                        
                        plt.plot(y_pred_df_gemval_1y2['lower value'], linestyle = '--', color = 'red', linewidth = 0.5,label='lower ci')
                        plt.plot(y_pred_df_gemval_1y2['upper value'], linestyle = '--', color = 'red', linewidth = 0.5,label='upper ci')
                        plt.fill_between(y_pred_df_gemval_1y2["Predictions"].index.values,
                                         y_pred_df_gemval_1y2['upper value'], 
                                         color = 'grey', alpha = 0.2)
                        plt.legend(loc = 'lower left', fontsize = 12)
                        st.pyplot(fig)
                        st.write(gemval_ar1_2_1y)
                    elif Prediction_Period=="2y":
                        y_gemval_2y,y_pred_df_gemval_2y2,gemval_ar1_2_2y=scfi_gemval.gemval_arima_2years(gemval)
                        st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 2 years</h5>", unsafe_allow_html=True)
                        fig = plt.figure(figsize = (16,8))
                        ax1 = fig.add_subplot(1, 1, 1)
                        ax1.set_facecolor('#EAF2F8')
                        plt.title("Confidence Interval after 2 years")
                        plt.plot(y_gemval_2y)
                        plt.plot(y_pred_df_gemval_2y2["Predictions"],label='predicted')
                        
                        plt.plot(y_pred_df_gemval_2y2['lower value'], linestyle = '--', color = 'red', linewidth = 0.5,label='lower ci')
                        plt.plot(y_pred_df_gemval_2y2['upper value'], linestyle = '--', color = 'red', linewidth = 0.5,label='upper ci')
                        plt.fill_between(y_pred_df_gemval_2y2["Predictions"].index.values,
                                         y_pred_df_gemval_2y2['upper value'], 
                                         color = 'grey', alpha = 0.2)
                        plt.legend(loc = 'lower left', fontsize = 12)
                        st.pyplot(fig)
                        st.write(gemval_ar1_2_2y)
                    
                elif Models=="LSTM":
                    if Prediction_Period=="7 Days":
                        st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Data set has weekly data values so can't predict</h5>", unsafe_allow_html=True) 
                    if Prediction_Period=="30 Days":
                        st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Data set has less data values so can't predict</h5>", unsafe_allow_html=True) 
                    if Prediction_Period=="180 Days":
                        y_test,pred_gemval_6m,eval=scfi_gemval.LSTM_6months(gemval)
                        st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 180 days</h5>", unsafe_allow_html=True)
                        fig = plt.figure(figsize = (16,8))
                        ax1 = fig.add_subplot(1, 1, 1)
                        ax1.set_facecolor('#EAF2F8')
                 
                        plt.plot(y_test,label='actual')
                        plt.plot(pred_gemval_6m,label='predicted')
                        
                        
                        plt.legend(loc = 'lower left', fontsize = 12)
                        st.pyplot(fig)
                        st.markdown("<h5 style='text-align: Left; color:  #EC7063;'>Model Performance</h5>", unsafe_allow_html=True)
                        st.write(eval)
                    elif Prediction_Period=="1y":
                        y_test,pred_gemval_1y,eval=scfi_gemval.LSTM_1year(gemval)
                        st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 1 year</h5>", unsafe_allow_html=True)
                        fig = plt.figure(figsize = (16,8))
                        ax1 = fig.add_subplot(1, 1, 1)
                        ax1.set_facecolor('#EAF2F8')
                  
                        plt.plot(y_test,label='actual')
                        plt.plot(pred_gemval_1y,label='predicted')
                        
                        
                        plt.legend(loc = 'lower left', fontsize = 12)
                        st.pyplot(fig)
                        st.markdown("<h5 style='text-align: Left; color:  #EC7063;'>Model Performance</h5>", unsafe_allow_html=True)
                        st.write(eval)
                    elif Prediction_Period=="2y":
                        y_test,pred_gemval_2y,eval=scfi_gemval.LSTM_2years(gemval)
                        st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 2 years</h5>", unsafe_allow_html=True)
                        fig = plt.figure(figsize = (16,8))
                        ax1 = fig.add_subplot(1, 1, 1)
                        ax1.set_facecolor('#EAF2F8')
                   
                        plt.plot(y_test,label='actual')
                        plt.plot(pred_gemval_2y,label='predicted')
                        
                        
                        plt.legend(loc = 'lower left', fontsize = 12)
                        st.pyplot(fig)
                        st.markdown("<h5 style='text-align: Left; color:  #EC7063;'>Model Performance</h5>", unsafe_allow_html=True)
                        st.write(eval)
                elif Models=="EXPO":
                    if Prediction_Period=="7 Days":
                        st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Data set has monthly so can't predict</h5>", unsafe_allow_html=True) 
                    if Prediction_Period=="30 Days":
                        st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Data set has monthly data values so can't predict</h5>", unsafe_allow_html=True) 
                    if Prediction_Period=="180 Days":
                            gemval_test_6m,gemval_pred1_6m,gemval_train_6m,model_performance =scfi_gemval.EXPO_6months(gemval)
                            st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 180 days</h5>", unsafe_allow_html=True)
                            fig = plt.figure(figsize = (16,8))
                            ax1 = fig.add_subplot(1, 1, 1)
                            ax1.set_facecolor('#EAF2F8')
               
                            plt.plot(gemval_train_6m, label='Train')
                            plt.plot(gemval_test_6m, gemval_pred1_6m, label='Exponential Smoothing ')
                            plt.legend(loc = 'lower left', fontsize = 12)
                            st.pyplot(fig)
                            st.markdown("<h5 style='text-align: Left; color:  #EC7063;'>Model Performance</h5>", unsafe_allow_html=True)
                            st.write(model_performance)
                 
                    elif Prediction_Period=="1y":
                        gemval_test_1y,gemval_pred1_1y,gemval_train_1y,model_performance =scfi_gemval.EXPO_1y(gemval)
                        st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 1 year</h5>", unsafe_allow_html=True)
                        fig = plt.figure(figsize = (16,8))
                        ax1 = fig.add_subplot(1, 1, 1)
                        ax1.set_facecolor('#EAF2F8')
                  
                        plt.plot(gemval_train_1y, label='Train')
                        plt.plot(gemval_test_1y, gemval_pred1_1y, label='Exponential Smoothing ')
                        plt.legend(loc = 'lower left', fontsize = 12)
                        st.pyplot(fig)
                        st.markdown("<h5 style='text-align: Left; color:  #EC7063;'>Model Performance</h5>", unsafe_allow_html=True)
                        st.write(model_performance)
                    elif Prediction_Period=="2y":
                        gemval_test_2y,gemval_pred1_2y,gemval_train_2y,model_performance=scfi_gemval.EXPO_2y(gemval)
                        st.markdown("<h5 style='text-align: center; color:  #EC7063;'>Forecast Predictions next 2 years</h5>", unsafe_allow_html=True)
                        fig = plt.figure(figsize = (16,8))
                        ax1 = fig.add_subplot(1, 1, 1)
                        ax1.set_facecolor('#EAF2F8')
               
                        plt.plot(gemval_train_2y, label='Train')
                        plt.plot(gemval_test_2y, gemval_pred1_2y, label='Exponential Smoothing ')
                        plt.legend(loc = 'lower left', fontsize = 12)
                        st.pyplot(fig)
                        st.markdown("<h5 style='text-align: Left; color:  #EC7063;'>Model Performance</h5>", unsafe_allow_html=True)
                        st.write(model_performance)
        
        
