from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
import pandas as pd

class DataHandler:
    @staticmethod
    def download_data(tickers, start_date, end_date):
        df = YahooDownloader(start_date, end_date, tickers).fetch_data()
        return df
        
    @staticmethod
    def preprocess_data(df, tech_indicator_list):
        fe = FeatureEngineer(use_technical_indicator=True, 
                            tech_indicator_list=tech_indicator_list,
                            use_turbulence=True)
        return fe.preprocess_data(df)
