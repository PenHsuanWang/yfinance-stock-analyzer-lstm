import os
import pandas as pd

from pandas_datareader.data import DataReader
import yfinance as yf
yf.pdr_override()


class YahooFinanceDataLoader:
    """
    define a class to load data from yahoo finance
    provide the start date and end date when initialize the class
    """

    def __init__(self, start_date, end_date):
        self._start_date = start_date
        self._end_date = end_date

    def get_company_data(self, company_list: list) -> dict[str, pd.DataFrame]:
        """
        fetching the data from yahoo finance
        """
        list_of_data = {}
        for stock in company_list:
            list_of_data[stock] = yf.download(stock, self._start_date, self._end_date)

        return list_of_data

    def get_compamy_data_and_save_to_csv(self, company_list: list, path_to_save_data: str) -> None:
        """
        fetching the data from yahoo finance and save to csv file
        """
        list_of_data = {}
        for stock in company_list:
            list_of_data[stock] = yf.download(stock, self._start_date, self._end_date)
            try:
                list_of_data[stock].to_csv(f"{path_to_save_data}/{stock}.csv")
            except OSError:
                print(f"Folder not found! Please check the path {path_to_save_data} exist")
                print(os.getcwd())

        return None


class CsvLoader:
    """
    define a class to load data from csv file
    provide the path of csv file when initialize the class
    """

    def __init__(self, csv_file_path_list: list[str]):
        self._csv_file_path = csv_file_path_list

    def get_data(self, company_name: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        get data from csv file
        provide company name, start date and end date to filter the desired data and return in pandas data frame format
        """

        for csv_file_path in self._csv_file_path:
            if csv_file_path.split(".")[-2].split("/")[-1] == company_name:
                data = pd.read_csv(csv_file_path)
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)
                data = data.loc[start_date:end_date]

                return data

        raise ValueError(f"Company name {company_name} not found in the csv file")

    def get_data_many(self, start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
        """
        get all the data from list of csv file
        return a dictionary of data in pandas data frame format
        """

        data_dict = {}
        for csv_file_path in self._csv_file_path:
            try:
                data = pd.read_csv(csv_file_path)
            except OSError:
                print(f"Folder not found! Please check the path {csv_file_path} exist")
                print(os.getcwd())

            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            data = data.loc[start_date:end_date]
            data_dict[csv_file_path.split(".")[-2].split("/")[-1]] = data

        return data_dict


if __name__ == '__main__':
    # The tech stocks we'll use for this analysis
    tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

    # create YahooFinanceDataLoader object
    yf_data_loader = YahooFinanceDataLoader(start_date='2015-01-01', end_date='2020-01-01')
    yf_data_loader.get_company_data(company_list=tech_list)

    company_list = [AAPL, GOOG, MSFT, AMZN]
    company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]

    print(AAPL.head())
    print(AAPL.describe())
