# 1. Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from isolation_forest import *


# 2. Load and Prepare
df = pd.read_csv("data/train.csv")
df = df.head(300).copy() # Use .copy() to avoid SettingWithCopy warnings

# Rename columns to match what your functions expect
df = df.rename(columns={
    'Packet Length Mean': 'Packet_Length', 
    'Label': 'Anomaly_Scores'
})

# 3. Select Features
features = ['Packet_Length', 'Anomaly_Scores']
X = df[features].copy()

# 4. Build Isolation Forest
iForest = isolation_forest(X, n_trees=20, max_depth=10, subspace=128)


# 5. Evaluate Dataset
scores = []

for i in range(X.shape[0]):
    score = anomaly_score(X.iloc[[i]], iForest, 128)
    scores.append(score)


# 6. Add Scores
X['anomaly_score'] = scores


# 7. Detect Outliers
threshold = 0.6
X['anomaly'] = X['anomaly_score'] > threshold


# 8. Display Results
pd.set_option('display.max_rows', None)
print(X.head(200))


# 9. GLOBAL DASHBOARD
fig, axes = plt.subplots(2, 2, figsize=(16,10))


# 10. Scatter Plot
normal = X[X['anomaly'] == False]
anomaly = X[X['anomaly'] == True]

axes[0,0].scatter(normal['Packet_Length'], normal['Anomaly_Scores'], alpha=0.5, label='Normal')
axes[0,0].scatter(anomaly['Packet_Length'], anomaly['Anomaly_Scores'], alpha=1, label='Anomaly')
axes[0,0].set_title("Isolation Forest From Scratch")
axes[0,0].set_xlabel("Packet Length")
axes[0,0].set_ylabel("Anomaly Scores")
axes[0,0].legend()


# 11. PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X[['Packet_Length', 'Anomaly_Scores']])

pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
pca_df['anomaly'] = X['anomaly'].values

normal_pca = pca_df[pca_df['anomaly'] == False]
anomaly_pca = pca_df[pca_df['anomaly'] == True]

axes[0,1].scatter(normal_pca['PCA1'], normal_pca['PCA2'], c='blue', marker='x', alpha=0.7, label='Normal')
axes[0,1].scatter(anomaly_pca['PCA1'], anomaly_pca['PCA2'], c='red', alpha=0.7, label='Anomaly')
axes[0,1].set_title("PCA 2D Visualization")
axes[0,1].set_xlabel("PCA 1")
axes[0,1].set_ylabel("PCA 2")
axes[0,1].legend()


# 12. Path Length Visualization
normal_instance = X[X['anomaly'] == False].sample(1)
outlier_instance = X[X['anomaly'] == True].sample(1)

bars1 = evaluate_instance(outlier_instance, iForest)
bars2 = evaluate_instance(normal_instance, iForest)

barWidth = 0.3
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

axes[1,0].bar(r2, bars2, width=barWidth, label='Normal Sample')
axes[1,0].bar(r1, bars1, width=barWidth, label='Outlier')
axes[1,0].set_title("Path Length Comparison")
axes[1,0].set_xlabel("Trees")
axes[1,0].set_ylabel("Tree Depth")
axes[1,0].legend()


# 13. Metrics Panel
outlier_score = anomaly_score(X[['Packet_Length', 'Anomaly_Scores']].head(1), iForest, 156)
normal_score = anomaly_score(X[['Packet_Length', 'Anomaly_Scores']].sample(1), iForest, 156)

axes[1,1].axis('off')
axes[1,1].text(0.1, 0.7, f'Outlier Score : {outlier_score:.4f}', fontsize=14)
axes[1,1].text(0.1, 0.5, f'Normal Score : {normal_score:.4f}', fontsize=14)
axes[1,1].text(0.1, 0.3, f'Total Trees : 20', fontsize=14)
axes[1,1].text(0.1, 0.1, f'Dataset Size : {len(X)}', fontsize=14)
axes[1,1].set_title("Isolation Forest Metrics")


# 14. Final Layout
plt.tight_layout()
plt.show()