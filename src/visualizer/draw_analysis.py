
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


class DrawMovingAverage:

    def __init__(self):
        sns.set_style('whitegrid')
        plt.style.use("fivethirtyeight")

    def draw_moving_average(self, company_name: str, company_data: pd.DataFrame) -> None:
        """
        plot the moving average of each company
        """
        available_moving_average = []
        for column_name in company_data.columns:
            if column_name.startswith('MA'):
                available_moving_average.append(column_name)

        plt.figure(figsize=(12, 8))
        plt.plot(company_data['Adj Close'], label='Adj Close')
        for moving_average in available_moving_average:
            plt.plot(company_data[moving_average], label=moving_average.replace('_', ' '))
        # plt.plot(data_to_plot['MA_20_days'], label='MA 20 days')
        # plt.plot(data_to_plot['MA_50_days'], label='MA 50 days')
        # plt.plot(data_to_plot['MA_100_days'], label='MA 100 days')
        # plt.plot(data_to_plot['MA_200_days'], label='MA 200 days')
        plt.title(f"Moving Average of {company_name}")
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Adj Close Price', fontsize=18)
        plt.legend(loc='upper left')
        plt.show()
