
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="DryBean AI - Classification Dashboard",
    page_icon="🌱",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"

FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4"
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Gaussian Naive Bayes": "gaussian_nb.pkl",
    "Support Vector Machine": "svm.pkl"
}

SCALED_MODELS = [
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Support Vector Machine"
]

BEAN_CLASSES = [
    "BARBUNYA",
    "BOMBAY",
    "CALI",
    "DERMASON",
    "HOROZ",
    "SEKER",
    "SIRA"
]

# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>
[data-testid="stMetric"] {
    background-color: #E8F5E9;
    padding: 20px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: #263238 !important;
}

[data-testid="stMetricValue"] {
    color: #1B5E20 !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# 3. APPLICATION HEADER
# =====================================================

st.markdown(
    '<div class="main-title">🌱 DryBean AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Classification & Evaluation Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# 4. LOAD SAVED MODELS
# =====================================================

@st.cache_resource
def load_models():

    models = {}

    for model_name, filename in MODEL_FILES.items():

        model_path = MODEL_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {filename}"
            )

        models[model_name] = joblib.load(model_path)

    scaler_path = MODEL_DIR / "scaler.pkl"

    if not scaler_path.exists():
        raise FileNotFoundError("Scaler file not found")

    scaler = joblib.load(scaler_path)

    return models, scaler


try:

    models, scaler = load_models()

except Exception as error:

    st.error(f"Unable to load models: {error}")
    st.stop()

# =====================================================
# 5. DATASET UPLOAD
# =====================================================

st.header("📂 Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your Dry Bean test dataset",
    type=["csv"]
)

if uploaded_file is None:

    st.info(
        "Please upload test_data.csv to begin model evaluation."
    )

    st.stop()

try:

    df = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(f"Unable to read CSV file: {error}")
    st.stop()

# =====================================================
# 6. DATA VALIDATION
# =====================================================

required_columns = FEATURES + ["Class"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(f"Missing columns: {missing_columns}")
    st.stop()

if df.empty:

    st.error("Uploaded dataset is empty.")
    st.stop()

if df[required_columns].isnull().any().any():

    st.error("Dataset contains missing values.")
    st.stop()

if not all(
    pd.api.types.is_numeric_dtype(df[col])
    for col in FEATURES
):

    st.error("All input features must be numeric.")
    st.stop()

if not df["Class"].isin(BEAN_CLASSES).all():

    st.error("Unknown bean class found in target column.")
    st.stop()

if len(df) < 2:

    st.error("At least two test records are required.")
    st.stop()

X_test = df[FEATURES]
y_test = df["Class"]

st.success("Dataset uploaded and validated successfully!")

# =====================================================
# 7. DATASET OVERVIEW
# =====================================================

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Input Features", len(FEATURES))
col3.metric("Bean Categories", y_test.nunique())

with st.expander("View Uploaded Dataset"):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

# =====================================================
# 8. MODEL EVALUATION FUNCTION
# =====================================================

def evaluate_model(model_name):

    model = models[model_name]

    # Apply scaling only to models that require it

    if model_name in SCALED_MODELS:

        X_input = scaler.transform(X_test)

    else:

        X_input = X_test

    # Generate predictions

    predictions = model.predict(X_input)

    # Generate class probabilities

    probabilities = model.predict_proba(X_input)

    # Calculate six evaluation metrics

    metrics = {

        "Accuracy": accuracy_score(
            y_test, predictions
        ),

        "AUC Score": roc_auc_score(
            y_test,
            probabilities,
            multi_class="ovr",
            average="weighted",
            labels=model.classes_
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "MCC Score": matthews_corrcoef(
            y_test,
            predictions
        )
    }

    return metrics, predictions, model.classes_

# =====================================================
# 9. MODEL SELECTION
# =====================================================

st.divider()

st.header("🤖 Machine Learning Model")

selected_model = st.selectbox(
    "Select a Classification Model",
    list(MODEL_FILES.keys())
)

try:

    metrics, predictions, classes = evaluate_model(
        selected_model
    )

except Exception as error:

    st.error(f"Model evaluation failed: {error}")
    st.stop()

# =====================================================
# 10. DISPLAY EVALUATION METRICS
# =====================================================

st.header("📈 Model Performance Metrics")

metric_cols = st.columns(3)

for index, (metric_name, metric_value) in enumerate(
    metrics.items()
):

    metric_cols[index % 3].metric(
        metric_name,
        f"{metric_value:.4f}"
    )

# =====================================================
# 11. CONFUSION MATRIX
# =====================================================

st.divider()

st.header("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    predictions,
    labels=classes
)

fig, ax = plt.subplots(figsize=(9, 6))

image = ax.imshow(cm, cmap="YlGnBu")

ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))

ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticklabels(classes)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")

ax.set_title(
    f"{selected_model} - Confusion Matrix"
)

for i in range(len(classes)):

    for j in range(len(classes)):

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="black"
        )

fig.colorbar(image, ax=ax)

fig.tight_layout()

st.pyplot(fig)

plt.close(fig)

# =====================================================
# 12. CLASSIFICATION REPORT
# =====================================================

st.header("Classification Report")

report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)

# =====================================================
# 13. COMPARE ALL FIVE MODELS
# =====================================================

st.divider()

st.header("🏆 Compare All Five Models")

if st.button("Compare All Models", type="primary"):

    comparison_results = []

    for model_name in MODEL_FILES:

        result, _, _ = evaluate_model(model_name)

        row = {
            "ML Model": model_name
        }

        row.update(result)

        comparison_results.append(row)

    comparison_df = pd.DataFrame(
        comparison_results
    )

    st.subheader("Model Performance Comparison")

    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    best_model = comparison_df.loc[
        comparison_df["Accuracy"].idxmax()
    ]

    st.success(
        f"Best Model: {best_model['ML Model']} "
        f"| Accuracy: {best_model['Accuracy']:.4f}"
    )

# =====================================================
# 14. FOOTER
# =====================================================

st.divider()

st.caption(
    "DryBean AI | M.Tech AIML | Machine Learning Assignment 2"
)
