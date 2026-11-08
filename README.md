# DryBean AI – Machine Learning Classification Dashboard

## 1. Project Overview

DryBean AI is a machine learning classification project developed to classify dry bean varieties using their physical characteristics.

The project uses five supervised machine learning algorithms and evaluates their performance using multiple classification metrics.

A Streamlit web application is developed to upload test datasets, select classification models, visualize results and compare model performance.

## 2. Problem Statement

The objective of this project is to develop a machine learning-based classification system that can accurately identify different varieties of dry beans using their physical characteristics.

Five supervised classification algorithms are implemented and evaluated to identify the most suitable model for predicting dry bean varieties. An interactive Streamlit dashboard is developed to enable users to upload test datasets, evaluate model performance, visualize classification results and compare the performance of different algorithms.


## 2.1 Dataset Information

- Dataset: Dry Bean Dataset
- Total test records: 2,723
- Input features: 16
- Target variable: Class
- Number of bean varieties: 7

  
### Dataset Source

The Dry Bean Dataset is obtained from the UCI Machine Learning Repository.

Dataset Reference:
https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

The dataset contains physical measurements of dry bean samples belonging to seven different varieties.


### Bean Categories

1. BARBUNYA
2. BOMBAY
3. CALI
4. DERMASON
5. HOROZ
6. SEKER
7. SIRA

## 3. Machine Learning Algorithms

The following five classification algorithms are implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 4. Evaluation Metrics

The models are evaluated using six performance metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Matthews Correlation Coefficient (MCC)


## 5. Model Performance Comparison

All five classification models were evaluated using the same test dataset containing 2,723 records.

The following six evaluation metrics were calculated:

| Model                 | Accuracy | ROC-AUC | Precision | Recall | F1 Score | MCC    |
|-----------------------|---------:|--------:|----------:|-------:|---------:|-------:|
| Logistic Regression   | 0.9214   | 0.9934  | 0.9222    | 0.9214 | 0.9216   | 0.9050 |
| Decision Tree         | 0.8917   | 0.9330  | 0.8912    | 0.8917 | 0.8913   | 0.8691 |
| KNN                   | 0.9166   | 0.9812  | 0.9174    | 0.9166 | 0.9168   | 0.8992 |
| Gaussian Naive Bayes  | 0.7639   | 0.9644  | 0.7654    | 0.7639 | 0.7615   | 0.7154 |
| Support Vector Machine| 0.9221   | 0.9889  | 0.9224    | 0.9221 | 0.9222   | 0.9059 |

Note: Precision, Recall and F1 Score are weighted averages for the multiclass classification problem.

### 5.1 Model Performance Observations

**1. Logistic Regression**

Logistic Regression achieved an accuracy of 92.14% and the highest ROC-AUC score of 0.9934. It demonstrated excellent class discrimination and strong overall classification performance.

**2. Decision Tree**

Decision Tree achieved an accuracy of 89.17%. It performed particularly well in classifying BOMBAY beans but showed comparatively weaker performance in identifying SIRA beans.

**3. K-Nearest Neighbors (KNN)**

KNN achieved an accuracy of 91.66% and ROC-AUC of 0.9812. It demonstrated strong classification performance, with particularly good results for BOMBAY, HOROZ and SEKER varieties.

**4. Gaussian Naive Bayes**

Gaussian Naive Bayes achieved an accuracy of 76.39%, which was the lowest among the five models. Although its ROC-AUC was 0.9644, its lower accuracy and MCC indicate comparatively weaker overall classification performance.

**5. Support Vector Machine (SVM)**

Support Vector Machine achieved the highest accuracy of 92.21%, with precision of 92.24%, recall of 92.21% and F1 Score of 92.22%. Its MCC score of 0.9059 also indicates strong classification performance.

### 5.2 Overall Best Performing Model

Support Vector Machine (SVM) achieved the highest classification accuracy of 92.21%, making it the overall winner based on accuracy.

Logistic Regression achieved the highest ROC-AUC score of 0.9934, indicating excellent class discrimination.

Both models demonstrated strong performance on the Dry Bean Dataset.


## 6. Project Links

### GitHub Repository
[DryBean Classification – Source Code](https://github.com/rajasekarsubhiksha/DryBean_Classification)

### Live Streamlit Application
[DryBean AI – Machine Learning Classification Dashboard](https://drybeanclassification-eiynkc8msipb8j8ff7mxw6.streamlit.app/)


## 7. Application Features

- CSV dataset upload
- Dataset validation
- Five-model selection
- Performance metric visualization
- Confusion matrix
- Classification report
- Comparison of all five models

## 8. Project Structure

```
DryBean_Classification/
│
├── 01_Dataset_Exploration.ipynb
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── gaussian_nb.pkl
    ├── svm.pkl
    └── scaler.pkl
```

## 9. Installation and Execution

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Upload the test_data.csv file to evaluate the trained classification models.

## 10. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit
- Jupyter Notebook

## 11. Conclusion

This project successfully implemented five supervised machine learning classification algorithms to classify seven varieties of dry beans using their physical characteristics.

The models were evaluated using Accuracy, ROC-AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

Among the five algorithms, Support Vector Machine achieved the highest accuracy of 92.21%, while Logistic Regression achieved the highest ROC-AUC of 0.9934.

An interactive Streamlit application was developed to facilitate dataset uploading, model selection, performance evaluation, confusion matrix visualization and classification report generation.

This project demonstrates an end-to-end machine learning workflow, covering model development, evaluation, GitHub version control and cloud deployment.
