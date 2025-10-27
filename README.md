## 🌍 Country Data Analysis and Clustering
### 📘 Overview

This project performs Exploratory Data Analysis (EDA), data transformation, and unsupervised clustering on a dataset containing country-level statistics. It also includes a Random Forest regression model to predict income based on socio-economic features.

## App:


## 📂 Features Covered

EDA

Missing value check

Summary statistics

Correlation heatmap

Data Preprocessing

Label encoding of categorical data

Skewness check and Power Transformation (Yeo-Johnson)

Outlier visualization using boxplots

Feature Scaling & PCA (optional)

Standardization using StandardScaler

Regression Model

Target: income

Model: RandomForestRegressor

Evaluation: R², MAE, MSE

Clustering Algorithms

K-Means

Agglomerative Clustering

DBSCAN

Metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz

Visualization

Cluster visualization for K-Means

Skewness comparison before and after transformation

## 🧠 Technologies Used
Category	Tools
Language	Python
Libraries	pandas, numpy, seaborn, matplotlib
ML/Clustering	scikit-learn
Model Saving	pickle
Visualization	seaborn, matplotlib

## ⚙️ Installation

Clone the repository:

git clone https://github.com/your-username/country-data-clustering.git
cd country-data-clustering


Create a virtual environment and install dependencies:

python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt

## 🧩 Usage
🏗️ Run the Jupyter Notebook
jupyter notebook task.ipynb

💻 Or Run Streamlit App

If you have an app.py file for model deployment:

streamlit run app.py

## 📊 Output Examples
🔹 Correlation Heatmap

Displays relationships between all numeric features after encoding.

🔹 Skewness Before vs After Transformation

Shows how Power Transformation improves feature distribution.

🔹 Regression Model Evaluation
Metric	Value
R²	~0.85
MAE	~0.04
MSE	~0.006
🔹 Clustering Results
Model	Silhouette	Davies-Bouldin	Calinski-Harabasz	Best
K-Means	0.56	0.72	140.5	✅
Agglomerative	0.48	0.81	120.3	
DBSCAN	0.32	1.12	85.7	
💾 Save & Load Model

Example code to save your trained Random Forest model:

import pickle

### Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

### Load
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

### 📈 Visualizations Included

Histogram and Boxplot (Before/After Power Transformation)
<img width="776" height="290" alt="image" src="https://github.com/user-attachments/assets/04be1b27-9094-4096-baf6-676e3a5e1c7b" />
<img width="790" height="290" alt="image" src="https://github.com/user-attachments/assets/111a4b0d-51c8-49a7-adcb-f29d1159657b" />

Correlation Heatmap
<img width="762" height="590" alt="image" src="https://github.com/user-attachments/assets/6abc9400-4dd5-4dd4-8e2c-8f200397987b" />

Cluster Visualization (K-Means)
<img width="590" height="390" alt="image" src="https://github.com/user-attachments/assets/615bdbd0-6839-4f1f-986f-ffb42ce65fee" />

Skewness Comparison Bar Chart
<img width="790" height="390" alt="image" src="https://github.com/user-attachments/assets/92625e07-ac8c-4c69-96c8-1c32eab16b68" />

## 🏁 Results

Power Transformation effectively reduced feature skewness.

K-Means performed best among the three clustering algorithms (highest silhouette score).

Random Forest regression produced a strong predictive performance for income.
