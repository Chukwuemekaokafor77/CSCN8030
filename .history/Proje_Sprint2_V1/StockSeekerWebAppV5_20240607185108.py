# StockSeekerWebAppV3.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from chaikin_oscillator import calculate_chaikin_oscillator  # Importing the new module

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Function to call local CSS sheet
def local_css(file_name):
    with open(file_name) as f:
        st.sidebar.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Provide the path to the style.css file
style_css_path = r"C:\Users\Admin\Documents\MLAI\CSCN8030\Proje_Sprint2\docs\assets\style.css"
local_css(style_css_path)

# Predefined list of popular stock tickers
popular_tickers = ['AAPL', 'META', 'NVDA', 'NFLX']

# Stock tickers combo box
st.sidebar.subheader("Stock Search Web App")
selected_stocks = st.sidebar.multiselect("Select stock tickers...", popular_tickers)

# Date range selection
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.now())

# Analysis type selection
analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Closing Prices", "Volume", "Moving Averages", "Daily Returns", "Correlation Heatmap", "Distribution of Daily Changes", "Predicted Prices", "Chaikin Oscillator"])

# Display additional information based on user selection
st.sidebar.subheader("Display Additional Information")
selected_options = {
    "Stock Actions": st.sidebar.checkbox("Stock Actions"),
    "Quarterly Financials": st.sidebar.checkbox("Quarterly Financials"),
    "Institutional Shareholders": st.sidebar.checkbox("Institutional Shareholders"),
    "Quarterly Balance Sheet": st.sidebar.checkbox("Quarterly Balance Sheet"),
    "Quarterly Cashflow": st.sidebar.checkbox("Quarterly Cashflow"),
    "Analysts Recommendation": st.sidebar.checkbox("Analysts Recommendation"),
    "Predicted Prices": st.sidebar.checkbox("Predicted Prices")  # Added prediction as a checkbox
}

# Submit button
button_clicked = st.sidebar.button("Analyze")

# Summary button
summary_clicked = st.sidebar.button("Show Technical Summary")

# Function to handle analysis
def handle_analysis(selected_stock, analysis_type, start_date, end_date):
    if analysis_type == "Predicted Prices":
        display_predicted_prices(selected_stock, start_date, end_date)
    elif analysis_type == "Chaikin Oscillator":
        display_chaikin_oscillator(selected_stock, start_date, end_date)
    else:
        display_stock_analysis(selected_stock, analysis_type, start_date, end_date)
        display_additional_information(selected_stock, selected_options)

# Function to display stock analysis
def display_stock_analysis(selected_stock, analysis_type, start_date, end_date):
    stock_data = yf.Ticker(selected_stock)
    stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
    st.subheader(f"{selected_stock} - {analysis_type}")

    if analysis_type == "Closing Prices":
        fig = px.line(stock_df, x=stock_df.index, y='Close', title=f'{selected_stock} Closing Prices')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Price')
        st.plotly_chart(fig)
        
    elif analysis_type == "Volume":
        fig = px.line(stock_df, x=stock_df.index, y='Volume', title=f'{selected_stock} Volume')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Volume')
        st.plotly_chart(fig)
        
    elif analysis_type == "Moving Averages":
        stock_df['MA20'] = stock_df['Close'].rolling(window=20).mean()
        stock_df['MA50'] = stock_df['Close'].rolling(window=50).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['MA20'], mode='lines', name='20-Day MA'))
        fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['MA50'], mode='lines', name='50-Day MA'))
        fig.update_layout(title=f'{selected_stock} Moving Averages',
                          xaxis_title='Date',
                          yaxis_title='Price')
        st.plotly_chart(fig)
        
    elif analysis_type == "Daily Returns":
        stock_df['Daily Return'] = stock_df['Close'].pct_change()
        fig = px.line(stock_df, x=stock_df.index, y='Daily Return', title=f'{selected_stock} Daily Returns')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Daily Return')
        st.plotly_chart(fig)
        
    elif analysis_type == "Correlation Heatmap":
        df_selected_stocks = yf.download(selected_stocks, start=start_date, end=end_date)['Close']
        corr = df_selected_stocks.corr()
        fig = px.imshow(corr, title='Correlation Heatmap')
        st.plotly_chart(fig)
        
    elif analysis_type == "Distribution of Daily Changes":
        stock_df['Daily Change'] = stock_df['Close'].diff()
        fig = px.histogram(stock_df['Daily Change'].dropna(), nbins=50, title='Distribution of Daily Changes')
        st.plotly_chart(fig)

# Function to display additional information
def display_additional_information(selected_stock, selected_options):
    for option, checked in selected_options.items():
        if checked:
            st.subheader(f"{selected_stock} - {option}")
            if option == "Stock Actions":
                display_action = yf.Ticker(selected_stock).actions
                if not display_action.empty:
                    st.write(display_action)
                else:
                    st.write("No data available")
            elif option == "Quarterly Financials":
                display_financials = yf.Ticker(selected_stock).quarterly_financials
                if not display_financials.empty:
                    st.write(display_financials)
                else:
                    st.write("No data available")
            elif option == "Institutional Shareholders":
                display_shareholders = yf.Ticker(selected_stock).institutional_holders
                if not display_shareholders.empty:
                    st.write(display_shareholders)
                else:
                    st.write("No data available")
            elif option == "Quarterly Balance Sheet":
                display_balancesheet = yf.Ticker(selected_stock).quarterly_balancesheet
                if not display_balancesheet.empty:
                    st.write(display_balancesheet)
                else:
                    st.write("No data available")
            elif option == "Quarterly Cashflow":
                display_cashflow = yf.Ticker(selected_stock).quarterly_cashflow
                if not display_cashflow.empty:
                    st.write(display_cashflow)
                else:
                    st.write("No data available")
            elif option == "Analysts Recommendation":
                display_recommendation = yf.Ticker(selected_stock).recommendations
                if not display_recommendation.empty:
                    st.write(display_recommendation)
                else:
                    st.write("No data available")

# Function to display the Chaikin Oscillator
def display_chaikin_oscillator(selected_stock, start_date, end_date):
    stock_data = yf.Ticker(selected_stock)
    stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
    
    # Calculate the Chaikin Oscillator
    stock_df.reset_index(inplace=True)
    stock_df = calculate_chaikin_oscillator(stock_df)
    st.subheader(f"{selected_stock} - Chaikin Oscillator")
    
    # Plot the Chaikin Oscillator along with the close price
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_df['Date'], y=stock_df['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=stock_df['Date'], y=stock_df['ch_osc'], mode='lines', name='Chaikin Oscillator'))
    fig.update_layout(title=f'{selected_stock} Chaikin Oscillator',
                      xaxis_title='Date',
                      yaxis_title='Price/Chaikin Oscillator')
    st.plotly_chart(fig)

# Function to display predicted prices
def display_predicted_prices(selected_stock, start_date, end_date):
    stock_data = yf.Ticker(selected_stock)
    stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
    st.subheader(f"{selected_stock} - Predicted Prices")

    # Prepare data for LSTM model
    df_close = stock_df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_close)

    prediction_days = 60
    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build LSTM model
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=1, batch_size=1)

    # Prepare test data
    test_start = start_date
    test_end = end_date

    test_data = stock_data.history(period='1d', start=test_start, end=test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((df_close, test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get predicted prices
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Plot the actual and predicted prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_data.index, y=actual_prices, mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(x=test_data.index, y=predicted_prices.flatten(), mode='lines', name='Predicted Price'))
    fig.update_layout(title=f'{selected_stock} - Predicted Prices',
                      xaxis_title='Date',
                      yaxis_title='Price')
    st.plotly_chart(fig)

# Main application logic
if button_clicked:
    for selected_stock in selected_stocks:
        handle_analysis(selected_stock, analysis_type, start_date, end_date)
elif summary_clicked:
    st.write("Technical Summary for selected stocks will be displayed here.")
else:
    st.write("Select stocks and analysis options from the sidebar to begin.")