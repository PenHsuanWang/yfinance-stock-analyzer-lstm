# This is the part of code to do stock price analysis
# define a class as StockAnalyzer
# input the data from YahooFinanceDataLoader in pandas data frame format

import pandas as pd


class StockPriceAnalyzer:

    def __init__(self, companies_data: dict[str, pd.DataFrame]):

        self._companies_data = companies_data

    def calculating_moving_average(self, window_size: int):
        """
        calculating the moving average of each company
        """
        for company in self._companies_data.values():
            column_name = f"MA_{window_size}_days"
            company[column_name] = company['Adj Close'].rolling(window_size).mean()
        return self

    def calculating_daily_return_percentage(self):
        """
        calculating the daily return percentage of each company
        """
        for company in self._companies_data.values():
            company['Daily Return'] = company['Adj Close'].pct_change()
        return self

    def get_analysis_data(self, company_name: str) -> pd.DataFrame:
        """
        get the analysis data of each company
        """
        return self._companies_data[company_name]


    
