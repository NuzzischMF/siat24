from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class PCAAnalyzer:
    def __init__(self, returns):
        self.returns = returns
        self.pca = None
        self.components = None
        self.explained_variance = None
        self.eigenportfolios = None
        
    def perform_pca(self):
        """
        Esegue PCA sui rendimenti standardizzati
        """
        # Standardizza i rendimenti
        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(self.returns)
        
        # Applica PCA
        self.pca = PCA()
        self.components = self.pca.fit_transform(returns_scaled)
        
        # Crea eigenportfolios
        self.eigenportfolios = pd.DataFrame(
            self.pca.components_,
            columns=self.returns.columns,
            index=[f'PC{i+1}' for i in range(len(self.returns.columns))]
        )
        
        self.explained_variance = self.pca.explained_variance_ratio_
        
        return self.eigenportfolios
    
    def analyze_variance(self):
        """
        Analizza la varianza spiegata
        """
        cumulative_var = np.cumsum(self.explained_variance)
        
        # Trova numero ottimale di componenti
        n_components_80 = np.where(cumulative_var >= 0.8)[0][0] + 1
        n_components_90 = np.where(cumulative_var >= 0.9)[0][0] + 1
        
        analysis = {
            'individual_variance': pd.Series(self.explained_variance),
            'cumulative_variance': pd.Series(cumulative_var),
            'n_components_80': n_components_80,
            'n_components_90': n_components_90
        }
        
        return analysis
    
    def get_factor_exposure(self, portfolio_weights):
        """
        Calcola l'esposizione di un portafoglio ai fattori PCA
        """
        return np.dot(portfolio_weights, self.eigenportfolios.T)
    
    def plot_results(self):
        """
        Crea visualizzazioni complete dell'analisi PCA
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Scree plot
        variance = self.analyze_variance()
        axes[0,0].plot(range(1, len(self.explained_variance) + 1),
                      self.explained_variance, 'bo-')
        axes[0,0].plot(range(1, len(self.explained_variance) + 1),
                      variance['cumulative_variance'], 'ro-')
        axes[0,0].set_title('Scree Plot')
        axes[0,0].set_xlabel('Principal Component')
        axes[0,0].set_ylabel('Explained Variance Ratio')
        axes[0,0].grid(True)
        
        # 2. First two eigenportfolios
        self.eigenportfolios.iloc[:2].T.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('First Two Eigenportfolios')
        axes[0,1].set_xlabel('Assets')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Heatmap of first 5 PCs
        sns.heatmap(self.eigenportfolios.iloc[:5].T, 
                   ax=axes[1,0], cmap='RdBu_r', center=0)
        axes[1,0].set_title('First 5 Principal Components')
        
        # 4. Biplot
        pc1 = self.components[:, 0]
        pc2 = self.components[:, 1]
        axes[1,1].scatter(pc1, pc2, alpha=0.5)
        axes[1,1].set_xlabel('First Principal Component')
        axes[1,1].set_ylabel('Second Principal Component')
        axes[1,1].set_title('PCA Biplot')
        
        for i, asset in enumerate(self.returns.columns):
            axes[1,1].annotate(
                asset,
                (self.pca.components_[0, i], self.pca.components_[1, i])
            )
            
        plt.tight_layout()
        return fig