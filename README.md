## Unsupervised Classification of Radar Echoes
This repository presents the Week 4 AI for Earth Observation. The task is to classify radar waveform echoes into sea ice and lead classes using an unsupervised learning approach, compute the average echo shape and waveform variability for each class, and evaluate the classification against ESA official labels using a confusion matrix.
The work builds on the provided notebook Unit_2_Unsupervised_Learning_ipynb, specifically within the Altimetry Classification section.

## 1. Echo Classification Using K-Means
Echo classification was performed using K-Means clustering with two clusters. The implementation is located in the section “Scatter Plots of Clustered Data” under Altimetry Classification.
The clustering code is:

```
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

labels = kmeans.labels_
```
The clustering results are visualised using three feature-space scatter plots generated in this section.
