import pandas as pd
from src.data_loader.data_fetcher import YahooFinanceDataLoader
from src.analyzer.pattern_recognizer import PatternRecognizer

def download_data():

    yfinance_data_loader = YahooFinanceDataLoader()
    yfinance_data_loader.get_company_data_and_save_to_csv(
        company_list=["AAPL"],
        start_date="2023-01-01",
        end_date="2023-10-01",
        path_to_save_data="./data/raw_data/"
    )


def main():
    data = pd.read_csv('./data/raw_data/AAPL.csv')
    recognizer = PatternRecognizer(data)
    labeled_data = recognizer.recognize_patterns()
    labeled_data.to_csv('./data/analyzed_data/patterned/labeled_AAPL.csv', index=False)
    print(labeled_data[['Pattern']].dropna())


if __name__ == "__main__":
    # download_data()
    main()

