# ============================================================
# UPI FRAUD DETECTION SYSTEM
# Streamlit Web Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="UPI Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("💳 UPI Fraud Detection System")

st.write(
    "Machine Learning based system for detecting "
    "fraudulent UPI transactions."
)

st.divider()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("UPI Fraud Detection_dataset.csv")

    return df


# Try to load dataset
try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "❌ Dataset not found!\n\n"
        "Please keep 'UPI Fraud Detection_dataset.csv' "
        "in the same folder as app.py."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Dataset",
        "Fraud Analysis",
        "Prediction"
    ]
)


# ============================================================
# FIND FRAUD TARGET COLUMN
# ============================================================

possible_targets = [
    "is_fraud",
    "fraud",
    "fraud_flag",
    "fraudulent",
    "isFraud",
    "Is_Fraud",
    "Fraud",
    "Is Fraud",
    "Fraud_Flag"
]

target_column = None

for column in possible_targets:

    if column in df.columns:

        target_column = column

        break


# If target not automatically found
if target_column is None:

    st.sidebar.warning(
        "⚠️ Fraud column not detected automatically."
    )

    target_column = st.sidebar.selectbox(
        "Select Fraud Target Column",
        df.columns
    )


# ============================================================
# TRAIN MODEL FUNCTION
# ============================================================

def train_model():

    # Make a copy
    data = df.copy()

    # Remove missing values
    data = data.dropna()

    # Separate features and target
    X = data.drop(
        columns=[target_column]
    )

    y = data[target_column]

    # --------------------------------------------------------
    # Encode categorical columns
    # --------------------------------------------------------

    categorical_columns = X.select_dtypes(
        include="object"
    ).columns

    for column in categorical_columns:

        encoder = LabelEncoder()

        X[column] = encoder.fit_transform(
            X[column].astype(str)
        )

    # --------------------------------------------------------
    # Encode target if categorical
    # --------------------------------------------------------

    if y.dtype == "object":

        target_encoder = LabelEncoder()

        y = target_encoder.fit_transform(
            y.astype(str)
        )

    # --------------------------------------------------------
    # Train-Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # Random Forest Model
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Prediction
    prediction = model.predict(
        X_test
    )

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        prediction
    )

    return (
        model,
        X.columns,
        accuracy,
        X_test,
        y_test,
        prediction
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Dashboard")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Transactions",
            len(df)
        )

    with col2:

        st.metric(
            "Total Features",
            len(df.columns)
        )

    with col3:

        st.metric(
            "Missing Values",
            int(
                df.isnull().sum().sum()
            )
        )

    with col4:

        st.metric(
            "Duplicate Rows",
            int(
                df.duplicated().sum()
            )
        )

    st.divider()

    # Dataset preview
    st.subheader(
        "📋 Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.divider()

    # Dataset information
    st.subheader(
        "📌 Dataset Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**Rows:**",
            df.shape[0]
        )

        st.write(
            "**Columns:**",
            df.shape[1]
        )

    with col2:

        st.write(
            "**Fraud Target:**",
            target_column
        )


# ============================================================
# DATASET PAGE
# ============================================================

elif page == "Dataset":

    st.header("📁 Complete Dataset")

    st.write(
        "Below is the complete UPI transaction dataset."
    )

    st.dataframe(
        df,
        use_container_width=True,
        height=600
    )

    st.divider()

    st.subheader(
        "📌 Column Names"
    )

    st.write(
        df.columns.tolist()
    )


# ============================================================
# FRAUD ANALYSIS PAGE
# ============================================================

elif page == "Fraud Analysis":

    st.header("🔍 Fraud Analysis")

    st.write(
        f"Fraud Target Column: **{target_column}**"
    )

    # Fraud distribution
    fraud_count = df[
        target_column
    ].value_counts()

    st.subheader(
        "Fraud vs Genuine Transactions"
    )

    st.bar_chart(
        fraud_count
    )

    st.divider()

    # Pie chart
    st.subheader(
        "📊 Transaction Distribution"
    )

    fig, ax = plt.subplots()

    ax.pie(
        fraud_count.values,
        labels=fraud_count.index,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title(
        "Fraud Distribution"
    )

    st.pyplot(fig)

    st.divider()

    st.subheader(
        "📋 Fraud Count"
    )

    result = pd.DataFrame({
        "Transaction Type":
            fraud_count.index,
        "Count":
            fraud_count.values
    })

    st.dataframe(
        result,
        use_container_width=True
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "Prediction":

    st.header(
        "🚨 UPI Fraud Prediction"
    )

    st.write(
        "Train the Machine Learning model "
        "and evaluate its fraud detection performance."
    )

    st.divider()

    # Train button
    if st.button(
        "🚀 Train Machine Learning Model"
    ):

        with st.spinner(
            "Training model... Please wait."
        ):

            try:

                (
                    model,
                    features,
                    accuracy,
                    X_test,
                    y_test,
                    prediction
                ) = train_model()

                st.success(
                    "✅ Model trained successfully!"
                )

                st.divider()

                # ------------------------------------------------
                # Accuracy
                # ------------------------------------------------

                st.subheader(
                    "🎯 Model Accuracy"
                )

                st.metric(
                    "Accuracy",
                    f"{accuracy * 100:.2f}%"
                )

                st.divider()

                # ------------------------------------------------
                # Classification Report
                # ------------------------------------------------

                st.subheader(
                    "📈 Classification Report"
                )

                report = classification_report(
                    y_test,
                    prediction,
                    output_dict=True
                )

                report_df = pd.DataFrame(
                    report
                ).transpose()

                st.dataframe(
                    report_df,
                    use_container_width=True
                )

                st.divider()

                # ------------------------------------------------
                # Confusion Matrix
                # ------------------------------------------------

                st.subheader(
                    "🔲 Confusion Matrix"
                )

                cm = confusion_matrix(
                    y_test,
                    prediction
                )

                fig, ax = plt.subplots()

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    ax=ax
                )

                ax.set_xlabel(
                    "Predicted"
                )

                ax.set_ylabel(
                    "Actual"
                )

                ax.set_title(
                    "Confusion Matrix"
                )

                st.pyplot(fig)

                st.divider()

                # ------------------------------------------------
                # Feature Importance
                # ------------------------------------------------

                st.subheader(
                    "⭐ Feature Importance"
                )

                importance = pd.Series(
                    model.feature_importances_,
                    index=features
                ).sort_values(
                    ascending=False
                )

                st.bar_chart(
                    importance.head(10)
                )

                st.divider()

                # ------------------------------------------------
                # Sample Predictions
                # ------------------------------------------------

                st.subheader(
                    "🔮 Sample Predictions"
                )

                result = pd.DataFrame({

                    "Actual":
                        y_test.values[:20],

                    "Predicted":
                        prediction[:20]

                })

                st.dataframe(
                    result,
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"❌ Error while training model:\n\n{e}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "UPI Fraud Detection System | "
    "Machine Learning + Streamlit"
)
