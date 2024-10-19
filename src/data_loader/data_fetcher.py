import os
import pandas as pd
from abc import ABC, abstractmethod
import yfinance as yf

import torch
from sklearn.preprocessing import MinMaxScaler

yf.pdr_override()


class BaseLoader(ABC):
    """
    define a abstraction class to load data
    """
    @abstractmethod
    def get_data_for_pytorch_model(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class YahooFinanceDataLoader:
    """
    define a class to load data from yahoo finance
    provide the start date and end date when initialize the class
    """

    @staticmethod
    def get_company_data(company_list: list, start_date, end_date) -> dict[str, pd.DataFrame]:
        """
        fetching the data from yahoo finance
        """
        list_of_data = {}
        for stock in company_list:
            list_of_data[stock] = yf.download(stock, start_date, end_date)

        return list_of_data

    @staticmethod
    def get_company_data_and_save_to_csv(company_list: list, start_date: str, end_date: str, path_to_save_data: str) -> None:
        """
        fetching the data from yahoo finance and save to csv file
        """
        for stock in company_list:
            data = yf.download(stock, start_date, end_date)
            try:
                data.to_csv(f"{path_to_save_data}/{stock}.csv")
            except OSError:
                print(f"Folder not found! Please check the path {path_to_save_data} exist")
                print(os.getcwd())


class CsvLoader:
    """
    define a class to load data from csv file
    """

    def __init__(self, csv_file_path_list: list[str]):
        self._csv_file_path = csv_file_path_list

    def get_data(self, company_name: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get data for a specific company from csv file within a date range
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
        Get data from all csv files in the list within a date range
        """
        data_dict = {}
        for csv_file_path in self._csv_file_path:
            company_name = csv_file_path.split(".")[-2].split("/")[-1]
            try:
                data = pd.read_csv(csv_file_path)
            except OSError:
                print(f"File not found! Please check the path {csv_file_path} exist")
                print(os.getcwd())
                continue

            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            data = data.loc[start_date:end_date]
            data_dict[company_name] = data

        return data_dict

    def prepare_training_data_for_pytorch_model(self, start_date: str, end_date: str) -> dict[str, [torch.Tensor, torch.Tensor]]:
        """
        Prepare data for PyTorch model training from csv files within a date range
        """
        data_dict = {}
        for csv_file_path in self._csv_file_path:
            company_name = csv_file_path.split(".")[-2].split("/")[-1]
            try:
                data = pd.read_csv(csv_file_path)
            except OSError:
                print(f"File not found! Please check the path {csv_file_path} exist")
                print(os.getcwd())
                continue

            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            data = data.loc[start_date:end_date]

            # scaled the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

            data_dict[company_name] = [torch.tensor(scaled_data), torch.tensor(data['Close'].values)]

        return data_dict


if __name__ == '__main__':
    # The tech stocks we'll use for this analysis
    tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

    # create YahooFinanceDataLoader object
    yf_data_loader = YahooFinanceDataLoader()
    df_list = yf_data_loader.get_company_data(company_list=tech_list, start_date='2023-01-01', end_date='2023-01-31')
    print(df_list)