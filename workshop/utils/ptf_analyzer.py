import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class PortfolioAnalyzer:
    def __init__(self, returns, pca_results):
        self.returns = returns
        self.pca_results = pca_results
        self.portfolio_weights = None
        self.performance_metrics = None
        self.risk_decomposition = None
        
    def construct_pca_portfolio(self, n_components=3, method='equal'):
        """
        Costruisce portfolio basato sui primi n componenti principali
        
        methods:
        - 'equal': peso uguale ai primi n componenti
        - 'variance': peso basato sulla varianza spiegata
        - 'risk_parity': equal risk contribution
        """
        if method == 'equal':
            # Usa primi n componenti con pesi uguali
            weights = np.zeros(len(self.returns.columns))
            for i in range(n_components):
                weights += self.pca_results.components_[i] / n_components
                
        elif method == 'variance':
            # Peso proporzionale alla varianza spiegata
            weights = np.zeros(len(self.returns.columns))
            var_ratios = self.pca_results.explained_variance_ratio_[:n_components]
            var_ratios = var_ratios / var_ratios.sum()
            
            for i in range(n_components):
                weights += self.pca_results.components_[i] * var_ratios[i]
                
        elif method == 'risk_parity':
            # Equal risk contribution dai componenti
            weights = self._risk_parity_weights(n_components)
            
        # Normalizza i pesi
        weights = weights / np.abs(weights).sum()
        self.portfolio_weights = pd.Series(weights, index=self.returns.columns)
        
        return self.portfolio_weights
    
    def _risk_parity_weights(self, n_components):
        """
        Implementa risk parity tra i primi n componenti
        """
        from scipy.optimize import minimize
        
        def risk_parity_objective(w, components, target_risk):
            portfolio_risk = np.sqrt(np.sum((w * components) ** 2, axis=1))
            return np.sum((portfolio_risk - target_risk) ** 2)
        
        components = self.pca_results.components_[:n_components]
        target_risk = 1.0 / n_components
        
        result = minimize(
            risk_parity_objective,
            x0=np.ones(n_components) / n_components,
            args=(components, target_risk),
            constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            bounds=[(0, 1) for _ in range(n_components)]
        )
        
        return result.x @ components
    
    def analyze_performance(self):
        """
        Calcola metriche di performance del portafoglio
        """
        if self.portfolio_weights is None:
            raise ValueError("Costruisci prima il portafoglio")
            
        portfolio_returns = self.returns @ self.portfolio_weights
        
        self.performance_metrics = {
            'Annual Return': portfolio_returns.mean() * 252,
            'Annual Volatility': portfolio_returns.std() * np.sqrt(252),
            'Sharpe Ratio': (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252),
            'Max Drawdown': self._calculate_max_drawdown(portfolio_returns),
            'Skewness': portfolio_returns.skew(),
            'Kurtosis': portfolio_returns.kurtosis(),
            'VaR 95%': portfolio_returns.quantile(0.05),
            'CVaR 95%': portfolio_returns[portfolio_returns <= portfolio_returns.quantile(0.05)].mean()
        }
        
        return self.performance_metrics
    
    def decompose_risk(self):
        """
        Decompone il rischio del portafoglio usando PCA
        """
        if self.portfolio_weights is None:
            raise ValueError("Costruisci prima il portafoglio")
            
        # Proiezione sui componenti principali
        pc_exposure = self.portfolio_weights @ self.pca_results.components_.T
        
        # Contributo al rischio per componente
        risk_contrib = pc_exposure ** 2 * self.pca_results.explained_variance_ratio_
        
        self.risk_decomposition = pd.Series(
            risk_contrib,
            index=[f'PC{i+1}' for i in range(len(risk_contrib))]
        )
        
        return self.risk_decomposition
    
    def plot_analysis(self):
        """
        Crea visualizzazioni complete dell'analisi del portafoglio
        """
        if self.portfolio_weights is None:
            raise ValueError("Costruisci prima il portafoglio")
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Portfolio Weights
        self.portfolio_weights.plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Portfolio Weights')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Cumulative Returns
        portfolio_returns = self.returns @ self.portfolio_weights
        cumulative_returns = (1 + portfolio_returns).cumprod()
        cumulative_returns.plot(ax=axes[0,1])
        axes[0,1].set_title('Cumulative Portfolio Returns')
        
        # 3. Risk Decomposition
        self.risk_decomposition.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Risk Decomposition by Principal Component')
        
        # 4. Rolling Metrics
        rolling_sharpe = (
            portfolio_returns.rolling(252).mean() / 
            portfolio_returns.rolling(252).std()
        ) * np.sqrt(252)
        
        rolling_sharpe.plot(ax=axes[1,1])
        axes[1,1].set_title('Rolling 1-Year Sharpe Ratio')
        
        plt.tight_layout()
        return fig
    
    def _calculate_max_drawdown(self, returns):
        """
        Calcola il maximum drawdown
        """
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        return drawdowns.min()
    
    def get_risk_metrics(self):
        """
        Calcola metriche di rischio addizionali
        """
        if self.portfolio_weights is None:
            raise ValueError("Costruisci prima il portafoglio")
            
        portfolio_returns = self.returns @ self.portfolio_weights
        
        # Calcola VaR usando diversi metodi
        var_metrics = {
            'Historical VaR 95%': np.percentile(portfolio_returns, 5),
            'Historical VaR 99%': np.percentile(portfolio_returns, 1),
            'Parametric VaR 95%': (portfolio_returns.mean() - 
                                  1.645 * portfolio_returns.std()),
            'Parametric VaR 99%': (portfolio_returns.mean() - 
                                  2.326 * portfolio_returns.std())
        }
        
        # Calcola misure di tail risk
        tail_metrics = {
            'Historical CVaR 95%': portfolio_returns[
                portfolio_returns <= np.percentile(portfolio_returns, 5)
            ].mean(),
            'Max Daily Loss': portfolio_returns.min(),
            'Worst Week': portfolio_returns.rolling(5).sum().min(),
            'Worst Month': portfolio_returns.rolling(21).sum().min()
        }
        
        return {**var_metrics, **tail_metrics}