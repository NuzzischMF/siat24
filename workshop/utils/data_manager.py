import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
from matplotlib import pyplot as plt

class DataManager:
    def __init__(self):
        self.data = None
        self.returns = None
        self.clean_returns = None
        
    def fetch_data(self, tickers, start_date='2018-01-01', end_date='2023-12-31'):
        """
        Scarica i dati da Yahoo Finance
        """
        data = {}
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date)
                data[ticker] = df['Adj Close']
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                
        self.data = pd.DataFrame(data)
        self.compute_returns()
        return self.data
    
    def compute_returns(self):
        """
        Calcola i rendimenti logaritmici
        """
        self.returns = np.log(self.data/self.data.shift(1))
        
    def clean_returns(self, zscore_threshold=3.0):
        """
        Pulisce i rendimenti da outlier
        """
        clean_rets = self.returns.copy()
        
        # Rimuovi outlier usando zscore
        z_scores = stats.zscore(clean_rets, nan_policy='omit')
        clean_rets[abs(z_scores) > zscore_threshold] = np.nan
        
        # Interpola valori mancanti
        clean_rets = clean_rets.interpolate(method='linear')
        
        self.clean_returns = clean_rets
        return clean_rets
    
    def quality_checks(self):
        """
        Esegue controlli di qualità sui dati
        """
        checks = {
            'missing_data': self.returns.isnull().sum(),
            'zero_returns': (self.returns == 0).sum(),
            'skewness': self.returns.skew(),
            'kurtosis': self.returns.kurtosis(),
            'jarque_bera': pd.DataFrame({
                'statistic': stats.jarque_bera(self.returns)[0],
                'p_value': stats.jarque_bera(self.returns)[1]
            })
        }
        return checks

    def plot_data_quality(self):
        """
        Visualizza metriche di qualità dei dati
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Returns distribution
        for col in self.returns.columns:
            sns.histplot(self.returns[col], kde=True, ax=axes[0,0])
        axes[0,0].set_title('Returns Distribution')
        
        # QQ Plot
        stats.probplot(self.returns.values.flatten(), dist="norm", plot=axes[0,1])
        axes[0,1].set_title('Normal Q-Q Plot')
        
        # Missing data
        self.returns.isnull().sum().plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Missing Values')
        
        # Correlation heatmap
        sns.heatmap(self.returns.corr(), ax=axes[1,1], cmap='RdBu_r')
        axes[1,1].set_title('Correlation Matrix')
        
        plt.tight_layout()
        return fig