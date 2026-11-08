# DryBean AI – Machine Learning Classification Dashboard

## 1. Project Overview

DryBean AI is a machine learning classification project developed to classify dry bean varieties using their physical characteristics.

The project uses five supervised machine learning algorithms and evaluates their performance using multiple classification metrics.

A Streamlit web application is developed to upload test datasets, select classification models, visualize results and compare model performance.

## 2. Dataset Information

- Dataset: Dry Bean Dataset
- Total test records: 2,723
- Input features: 16
- Target variable: Class
- Number of bean varieties: 7

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

| Model               | Accuracy | ROC-AUC |
|---------------------|---------:|--------:|
| Logistic Regression | 92.14%   | 99.34%  |
| Decision Tree       | 89.17%   | 93.30%  |
| KNN                 |  91.66%  | 98.12%  |
| Gaussian Naive Bayes| 76.39%   |  96.44% |
| SVM                 | 92.21%   | 98.89%  |

SVM achieved the highest classification accuracy, while Logistic Regression achieved the highest ROC-AUC.

## 6. Application Features

- CSV dataset upload
- Dataset validation
- Five-model selection
- Performance metric visualization
- Confusion matrix
- Classification report
- Comparison of all five models

## 7. Project Structure

```
DryBean_Classification/
│
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

## 8. Installation and Execution

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Upload the test_data.csv file to evaluate the trained classification models.

## 9. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit
- Jupyter Notebook

## 10. Conclusion

Five supervised machine learning classification algorithms were implemented and evaluated using the Dry Bean Dataset.

Among the evaluated models, Support Vector Machine achieved the highest accuracy of 92.21%, while Logistic Regression achieved the highest ROC-AUC of 0.9934.

The Streamlit application provides an interactive interface for model evaluation and comparison.