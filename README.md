## Unsupervised Classification of Radar Echoes
This repository presents the Week 4 AI for Earth Observation. The task is to classify radar waveform echoes into sea ice and lead classes using an unsupervised learning approach, compute the average echo shape and waveform variability for each class, and evaluate the classification against ESA official labels using a confusion matrix.
The work builds on the provided notebook Unit_2_Unsupervised_Learning_ipynb, specifically within the Altimetry Classification section.

## Getting Started
The analysis was conducted in Google Colab. To reproduce the results, open the notebook Week4_Unsupervised_Echo_Classification.ipynb and run all cells in order. All figures and evaluation outputs will be generated automatically.

## Installation
The analysis relies on standard scientific Python libraries. The required packages are:
```
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install scipy

```
If running locally instead of Google Colab, ensure that these libraries are installed before executing the notebook.
The dataset used in this project is not included in the repository due to size constraints. All waveform features and ESA labels are loaded directly within the notebook as provided in the course materials.

## Data and Feature Space
The dataset consists of radar waveform echoes derived from satellite altimetry measurements. From each waveform, features including backscatter coefficient (sig₀), peakiness parameter (PP), and stack standard deviation (SSD) were used to perform clustering.
These features form the input matrix for K-Means clustering.

## Methodology
K-Means Classification
Unsupervised classification was performed using K-Means clustering with two clusters:
```
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

labels = kmeans.labels_
```
This implementation is located in the section “Scatter Plots of Clustered Data” under Altimetry Classification.

## Clustering Results
The clustering output is visualised in feature space through the following diagrams generated in Section 3 of the notebook.
<img width="565" height="433" alt="sig_0_SSD" src="https://github.com/user-attachments/assets/2743a31c-9a57-48df-bd72-af3fc05e0ec5" />
<img width="565" height="433" alt="sig_0_PP" src="https://github.com/user-attachments/assets/86f8f8cb-cbc2-4a64-9662-c1249e75b59d" />
<img width="565" height="432" alt="PP_SSD" src="https://github.com/user-attachments/assets/95e2abc1-99f2-49d2-a952-4ed2154e025e" />
These plots show clear separation between two echo populations corresponding to sea ice and lead classes.

## Aggregate Waveform Analysis
The average echo shape and waveform variability were analysed in the section “Aggregate alignment comparison”.
The mean waveform per class was computed using:
```
mean_class0 = np.mean(wf[labels == 0], axis=0)
mean_class1 = np.mean(wf[labels == 1], axis=0)
```
Standard deviation of waveform amplitude was computed using:
```
std_class0 = np.std(wf[labels == 0], axis=0)
std_class1 = np.std(wf[labels == 1], axis=0)
```
The aggregate comparison figure is shown below.
<img width="1489" height="985" alt="Aggregate_comparison" src="https://github.com/user-attachments/assets/3fa2ad9b-be74-4702-8ecd-65c52958d6fb" />

The lower-right panel displays the average echo shape for sea ice and lead classes. The aligned lead waveform exhibits a sharper and higher peak, while the sea ice waveform is broader. The standard deviation quantifies waveform variability within each class.

## 3. Quantitative Comparison with ESA Official Classification
Classification performance was evaluated in the section “Compare with ESA data”. The predicted cluster labels were compared to ESA official labels using a confusion matrix.
The evaluation code is:
```
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(esa_labels, labels)
print("Confusion Matrix:")
print(cm)

print(classification_report(esa_labels, labels))
```
The confusion matrix obtained was:
[[8856 22]
[ 24 3293]]
The results show high precision, recall, and overall accuracy close to one, indicating strong agreement between unsupervised clustering and ESA official classification.



