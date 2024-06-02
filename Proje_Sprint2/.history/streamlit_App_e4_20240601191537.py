import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
# Set plotting styles
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# Function to call local CSS sheet
def local_css(file_name):
    with open(file_name) as f:
        st.sidebar.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Provide the path to the style.css file
style_css_path = r"C:\Users\Admin\Documents\MLAI\CSCN8030\proje7\style.css"
local_css(style_css_path)

# Predefined list of popular stock tickers
popular_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']

# Stock tickers combo box
st.sidebar.subheader("Stock Search Web App")
selected_stocks = st.sidebar.multiselect("Select stock tickers...", popular_tickers)

# Date range selection
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.now())

# Analysis type selection
analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Closing Prices", "Volume", "Moving Averages", "Daily Returns", "Correlation Heatmap", "Distribution of Daily Changes"])

# Display additional information based on user selection
st.sidebar.subheader("Display Additional Information")
selected_options = {
    "Stock Actions": st.sidebar.checkbox("Stock Actions"),
    "Quarterly Financials": st.sidebar.checkbox("Quarterly Financials"),
    "Institutional Shareholders": st.sidebar.checkbox("Institutional Shareholders"),
    "Quarterly Balance Sheet": st.sidebar.checkbox("Quarterly Balance Sheet"),
    "Quarterly Cashflow": st.sidebar.checkbox("Quarterly Cashflow"),
    "Analysts Recommendation": st.sidebar.checkbox("Analysts Recommendation"),
    "Predicted Prices": st.sidebar.checkbox("Predicted Prices")
}

# Submit button
button_clicked = st.sidebar.button("Analyze")

def display_stock_analysis(selected_stock, analysis_type, start_date, end_date):
    stock_data = yf.Ticker(selected_stock)
    stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
    st.subheader(f"{selected_stock} - {analysis_type}")
    
    if analysis_type == "Closing Prices":
        st.line_chart(stock_df['Close'])
    elif analysis_type == "Volume":
        st.line_chart(stock_df['Volume'])
    elif analysis_type == "Moving Averages":
        stock_df['MA20'] = stock_df['Close'].rolling(window=20).mean()
        stock_df['MA50'] = stock_df['Close'].rolling(window=50).mean()
        fig, ax = plt.subplots()
        ax.plot(stock_df.index, stock_df['Close'], label='Closing Prices')
        ax.plot(stock_df.index, stock_df['MA20'], label='20-Day MA')
        ax.plot(stock_df.index, stock_df['MA50'], label='50-Day MA')
        ax.set_title("Moving Averages")
        ax.legend()
        st.pyplot(fig)
    elif analysis_type == "Daily Returns":
        stock_df['Daily Return'] = stock_df['Close'].pct_change()
        st.line_chart(stock_df['Daily Return'])
    elif analysis_type == "Correlation Heatmap":
        df_selected_stocks = yf.download(selected_stocks, start=start_date, end=end_date)['Close']
        corr = df_selected_stocks.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    elif analysis_type == "Distribution of Daily Changes":
        stock_df['Daily Change'] = stock_df['Close'].diff()
        fig, ax = plt.subplots()
        sns.histplot(stock_df['Daily Change'].dropna(), bins=50, kde=True, ax=ax)
        ax.set_title("Distribution of Daily Changes")
        st.pyplot(fig)

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
                display_balancesheet = yf.Ticker(selected_stock).quarterly_balance_sheet
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
                display_analyst_rec = yf.Ticker(selected_stock).recommendations
                if not display_analyst_rec.empty:
                    st.write(display_analyst_rec)
                else:
                    st.write("No data available")
            elif option == "Predicted Prices":
                # Load the trained model
                model_path = r"C:\Users\Admin\Documents\MLAI\CSCN8030\proje7\AAPL_model.pkl"
                model = joblib.load(model_path)
                # Prepare data for prediction
                stock_data = yf.Ticker(selected_stock).history(period='1d', start=start_date, end=end_date)
                # Predict prices
                predicted_prices = model.predict(stock_data[['Open', 'High', 'Low', 'Close']])
                # Plot predicted and actual prices
                fig, ax = plt.subplots()
                ax.plot(stock_data.index, stock_data['Close'], label='Actual Prices')
                ax.plot(stock_data.index, predicted_prices, label='Predicted Prices')
                ax.set_title("Actual vs Predicted Prices")
                ax.legend()
                st.pyplot(fig)
            else:
                # Handle other options if needed
                pass
           

def main():
    if not selected_stocks:
        st.sidebar.warning("Please select at least one stock ticker.")
        return

    for selected_stock in selected_stocks:
        display_stock_analysis(selected_stock, analysis_type, start_date, end_date)
        display_additional_information(selected_stock, selected_options)

if __name__ == "__main__":
    main()