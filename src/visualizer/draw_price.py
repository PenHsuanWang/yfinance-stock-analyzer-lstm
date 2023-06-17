
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


class DrawPrice:

    def __init__(self, companies_data: dict[str, pd.DataFrame]):
        self._companies_data = companies_data
        sns.set_style('whitegrid')
        plt.style.use("fivethirtyeight")

    def plot_stock_price(self, company_name: str) -> None:
        """
        plot the stock price of each company
        """
        data_to_plot = self._companies_data[company_name]
        plt.figure(figsize=(12, 8))
        plt.plot(data_to_plot['Adj Close'])
        plt.title(f"Closing Price of {company_name}")
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Adj Close Price', fontsize=18)
        plt.show()

    def plot_many_companies_stock_price_in_one_plot(self, company_list: list) -> None:
        """
        plot the stock price of each company in one plot
        """
        plt.figure(figsize=(12, 8))
        for company in company_list:
            plt.plot(self._companies_data[company]['Adj Close'])
            # plt.plot(company['Adj Close'])
        plt.title("Adjusted Close Price")
        plt.legend(company_list, loc='upper left')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Adj Close Price', fontsize=18)
        plt.tight_layout()
        plt.show()

