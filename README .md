# Data Project - Stock Market Analysis

## Objective
This notebook explores stock market data, focusing on technology giants like Apple, Amazon, Google, and Microsoft. It demonstrates the use of yfinance to retrieve stock information and visualization techniques with Seaborn and Matplotlib. The analysis includes assessing stock risk using historical performance data and predicting future prices using a Linear Regression model.

## Stock Market Reactions to Election

## Task Breakdown
1. **Identify reliable market data APIs**
2. **Develop scripts/tools for data ingestion**
3. **Clean and preprocess collected data**
4. **Standardize data formats**
5. **Explore data visualization techniques**
6. **Perform exploratory data analysis (EDA)**
7. **Extract relevant features from raw financial data**
8. **Implement data transformation techniques**
9. **Split the preprocessed data into training, validation, and test sets**
10. **Document data collection and preprocessing procedures**
    - Keep thorough documentation of each step for reproducibility.

## Documentation

### Data Collection
- Data sourced from Yahoo Finance using yfinance library.
- Stock symbols: AAPL, GOOG, MSFT, AMZN.
- Time period: Last one year.

### Data Preprocessing
- Filled missing values using forward fill.
- Added company name column.
- Concatenated individual stock data into a single DataFrame.

### Data Standardization
- Ensured consistent date format.
- Handled missing values.

### Data Visualization
- Plotted closing prices and volume of sales.
- Calculated and plotted moving averages (10, 20, 50 days).
- Visualized daily returns using histograms and line plots.

### Feature Extraction
- Calculated daily returns.
- Analyzed correlations between stock returns using heatmaps and pair plots.

### Data Splitting
- Split data into training and test sets for model validation.

## Conclusion
In this notebook, we delved into the world of stock market data analysis. Here's a summary of what we explored:

- We learned how to retrieve stock market data from Yahoo Finance using the yfinance library.
- Using Pandas, Matplotlib, and Seaborn, we visualized time-series data to gain insights into the stock market trends.
- We measured the correlation between different stocks to understand how they move in relation to each other.
- We assessed the risk associated with investing in a particular stock by analyzing its daily returns.
- Lastly, we split the data into training and validation sets for further analysis and model training.

If you have any questions or need further clarification on any topic covered in this notebook, feel free to ask in the comments below. I'll be happy to assist you!

## References
- Investopedia on Correlation
- [Article 1](file:///C:/Users/Admin/Desktop/C_AIML/semestert2/AI%20for%20Business/article1.pdf)
- [Stock Data Analysis Project](https://medium.com/@ethan.duong1120/stock-data-analysis-project-python-1bf2c51b615f)
