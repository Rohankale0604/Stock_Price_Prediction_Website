# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 23:07:10 2024

@author: Rohan
"""

import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.statespace import sarimax

data = pd.read_csv(r"C:\Users\Rohan\Downloads\RELIANCE.NS Final.csv")

data["Date"] = pd.to_datetime(data["Date"])

data.set_index("Date", inplace=True)

st.set_option('deprecation.showPyplotGlobalUse', False)


st.title("Stock market Prediction of Reliance Industries")
navigation = st.sidebar.radio("Options",["Historic data","Visualization","Predictions"])

if navigation =="Historic data":
    st.write("Historic data from 1996 to 2024 Jan")
    image_path = r"C:\Users\sunfa\Downloads\stockpic.jpg"

    st.image(image_path, caption='Stock Market',use_column_width=True)
    
    if st.checkbox("Show Table"):
        st.table(data)
    
if navigation =="Visualization":
    st.write("Visuals")
    graph = st.selectbox("Type of Graph for visual",["Non Interactive","Interactive"])
    
    st.set_option('deprecation.showPyplotGlobalUse', False)


    val = st.slider("Filter the data in years",1996,2023)
    
    data_filtered = data[data.index > pd.to_datetime(str(val))]
  


    if graph == "Non Interactive":
        plt.figure(figsize = (10,5))
        plt.scatter(data_filtered.index, data_filtered["Close"])
        plt.xlabel("Years")
        plt.ylabel("Closing prices")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()


        
        
    if graph == "Interactive":
        layout = go.Layout( 
        xaxis=dict(range=[data_filtered.index.min(),data_filtered.index.max()]),
        yaxis=dict(range=[13, 3000])
        )
        fig = go.Figure(data=go.Scatter(x=data_filtered.index, y=data_filtered["Close"]), layout=layout)
        st.plotly_chart(fig)
        
        

    
if navigation == "Predictions":
    st.write("Future Predictions")
    
 
    input_days = st.number_input("Enter the number of days you want to make the prediction for", min_value=1, step=1)
    
    
    
    
    updated_model = sarimax.SARIMAX(data['Close'], order=(1, 1, 1), seasonal_order=(2, 1, 0, 12))
    updated_res = updated_model.fit()

    st.header("Stock prices of Reliance")
    
    forecast_dates = pd.date_range(data.index.max() + pd.DateOffset(1), periods=input_days)

    predictions = updated_res.get_forecast(steps=input_days)
    
    future_data = pd.DataFrame({
        'Date': forecast_dates,
        'Predicted_Close': predictions.predicted_mean
    })

    if st.button("Display predicted values"):
         st.write(future_data)
         
    pred2ci = predictions.conf_int()
    predictions.predicted_mean.plot(label='Predicted Close Prices')


    plt.fill_between(pred2ci.index, pred2ci.iloc[:, 0], pred2ci.iloc[:, 1], color='k', alpha=.1)

    plt.ylabel('Close price')
    plt.legend(loc='best')
    plt.title('Forecasting of Close price of stock Reliance with confidence interval')

   
        #st.success(f"The forecasted stock prices are: {future_data}")

# Display the plot
st.pyplot()
    