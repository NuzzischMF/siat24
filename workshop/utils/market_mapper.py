from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class MarketMapper:
    def __init__(self, returns):
        self.returns = returns
        self.scaled_returns = None
        self.tsne_map = None
        self.umap_map = None
        
    def prepare_data(self):
        """
        Standardizza i rendimenti per il mapping
        """
        scaler = StandardScaler()
        self.scaled_returns = scaler.fit_transform(self.returns)
        return self.scaled_returns
    
    def create_tsne_map(self, perplexity=30):
        """
        Crea mapping usando t-SNE
        """
        if self.scaled_returns is None:
            self.prepare_data()
            
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(self.returns.columns)-1),
            random_state=42
        )
        
        self.tsne_map = pd.DataFrame(
            tsne.fit_transform(self.scaled_returns.T),
            columns=['x', 'y'],
            index=self.returns.columns
        )
        
        return self.tsne_map
    
    def create_umap_map(self, n_neighbors=15, min_dist=0.1):
        """
        Crea mapping usando UMAP
        """
        if self.scaled_returns is None:
            self.prepare_data()
            
        umap = UMAP(
            n_neighbors=min(n_neighbors, len(self.returns.columns)-1),
            min_dist=min_dist,
            random_state=42
        )
        
        self.umap_map = pd.DataFrame(
            umap.fit_transform(self.scaled_returns.T),
            columns=['x', 'y'],
            index=self.returns.columns
        )
        
        return self.umap_map
    
    def plot_maps(self, figsize=(15, 6)):
        """
        Visualizza e confronta i mapping
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # t-SNE plot
        ax1.scatter(self.tsne_map['x'], self.tsne_map['y'])
        for idx in self.tsne_map.index:
            ax1.annotate(idx, (self.tsne_map.loc[idx, 'x'],
                             self.tsne_map.loc[idx, 'y']))
        ax1.set_title('t-SNE Market Map')
        
        # UMAP plot
        ax2.scatter(self.umap_map['x'], self.umap_map['y'])
        for idx in self.umap_map.index:
            ax2.annotate(idx, (self.umap_map.loc[idx, 'x'],
                             self.umap_map.loc[idx, 'y']))
        ax2.set_title('UMAP Market Map')
        
        plt.tight_layout()
        return fig
    
    def analyze_clusters(self, method='umap', n_clusters=5):
        """
        Analizza cluster nei mapping
        """
        from sklearn.cluster import KMeans
        
        map_data = self.umap_map if method == 'umap' else self.tsne_map
        
        # Applica clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(map_data)
        
        # Analizza composizione cluster
        cluster_composition = {}
        for i in range(n_clusters):
            assets = map_data.index[clusters == i]
            cluster_composition[f'Cluster_{i}'] = list(assets)
            
        return clusters, cluster_composition