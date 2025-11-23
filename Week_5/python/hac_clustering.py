# hac_clustering.py
# pip install numpy pandas scipy scikit-learn matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# Example: data/india_states_education.csv with numeric columns
data = pd.read_csv("data/india_states_education.csv", index_col=0)  # rows = states, cols = features
X = data.values
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# linkage and dendrogram
Z = linkage(Xs, method='ward')   # try 'single', 'complete', 'average', 'ward'
plt.figure(figsize=(10,6))
dendrogram(Z, labels=data.index, leaf_rotation=90, leaf_font_size=8)
plt.title("Hierarchical Clustering (Ward)")
plt.tight_layout()
plt.savefig("figures/fig_hac_dendrogram.png", dpi=150)
plt.close()
