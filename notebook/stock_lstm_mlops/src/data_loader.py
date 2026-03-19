import os
import pandas as pd
import yfinance as yf
from datetime import datetime

class StockDataLoader:
    def __init__(self, data_path: str, ticker: str = "AAPL", start_date: str = "2012-01-01", end_date: str = None):
        self.data_path = data_path
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')

    def load_data(self) -> pd.DataFrame:
        if os.path.exists(self.data_path):
            print(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path, index_col='Date', parse_dates=True)
        else:
            print(f"Downloading data for {self.ticker} from {self.start_date} to {self.end_date}")
            df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
            
            # yfinance returns a MultiIndex for columns in newer versions. Flatten it.
            if isinstance(df.columns, pd.MultiIndex):
                # Drop the 'Ticker' level if it exists
                if 'Ticker' in df.columns.names:
                    df.columns = df.columns.droplevel('Ticker')
                else:
                    # Fallback: flatten the multi-index
                    df.columns = ['_'.join(col).strip() for col in df.columns.values]
                    
            # Create directory if not exists
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            df.to_csv(self.data_path)
            print(f"Data saved to {self.data_path}")
            
        return df[['Close']]