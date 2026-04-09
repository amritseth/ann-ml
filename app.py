# ==============================

# 📊 Advertising Sales Predictor

# Developed by Amrit Seth

# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import metrics

# ------------------------------

# 🔧 Page Configuration

# ------------------------------

st.set_page_config(page_title="Sales Prediction Dashboard | Amrit Seth", layout="wide")

st.title("📊 Advertising Sales Prediction Dashboard")
st.markdown("""
👨‍💻 **Developed by Amrit Seth**

This project demonstrates a complete **Machine Learning pipeline** where I analyze how advertising budgets
(TV, Radio, Newspaper, etc.) affect product sales.

➡️ Upload your dataset or use the sample dataset to explore insights, train models, and predict sales.
""")

# ------------------------------

# 🧠 Session State Initialization

# ------------------------------

if 'df' not in st.session_state:
st.session_state.df = None
if 'processed_df' not in st.session_state:
st.session_state.processed_df = None
if 'model' not in st.session_state:
st.session_state.model = None
if 'scaler' not in st.session_state:
st.session_state.scaler = None

# ------------------------------

# 📂 Step 1: Data Input

# ------------------------------

with st.expander("📂 Step 1: Load Dataset", expanded=True):
import os
local_dataset_path = "dataset_files/Advertising Budget and Sales.csv"

```
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

with col2:
    use_sample = st.checkbox("Use Sample Dataset", value=os.path.exists(local_dataset_path))

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
elif use_sample and os.path.exists(local_dataset_path):
    df = pd.read_csv(local_dataset_path)
    st.success("✅ Sample dataset loaded!")
else:
    df = None

if df is not None:
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Auto-detect target
    target_candidates = ['Sales', 'sales']
    detected_target = next((c for c in target_candidates if c in df.columns), df.columns[-1])

    st.session_state.target_col = st.selectbox("🎯 Select Target Column", df.columns,
                                               index=df.columns.get_loc(detected_target))

    st.session_state.df = df

    st.write("### 🔍 Preview")
    st.dataframe(df.head())
    st.write(f"📌 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
```

# ------------------------------

# Continue if data exists

# ------------------------------

if st.session_state.df is not None:

```
df = st.session_state.df
target_col = st.session_state.target_col

# ------------------------------
# 🔍 Step 2: EDA
# ------------------------------
with st.expander("🔍 Step 2: Exploratory Data Analysis"):
    col1, col2 = st.columns(2)

    with col1:
        st.write("### 📊 Statistical Summary")
        st.write(df.describe())

    with col2:
        st.write("### 🔥 Correlation Heatmap")
        fig = px.imshow(df.corr(), text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

    features = [col for col in df.columns if col != target_col]

    selected_feature = st.selectbox("📈 Feature vs Target", features)

    fig2 = px.scatter(df, x=selected_feature, y=target_col, trendline="ols")
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------
# 🛠 Step 3: Data Processing
# ------------------------------
with st.expander("🛠 Step 3: Data Cleaning & Scaling"):
    if df.isnull().sum().sum() > 0:
        if st.button("Fill Missing Values"):
            df = df.fillna(df.mean())
            st.session_state.df = df
            st.success("Missing values handled!")

    scaling = st.radio("Scaling Method", ["None", "StandardScaler", "MinMaxScaler"])

    processed_df = df.copy()

    if scaling != "None":
        scaler = StandardScaler() if scaling == "StandardScaler" else MinMaxScaler()
        cols = [c for c in df.columns if c != target_col]
        processed_df[cols] = scaler.fit_transform(df[cols])
        st.session_state.scaler = scaler

    st.session_state.processed_df = processed_df

# ------------------------------
# 🎯 Step 4: Feature Selection
# ------------------------------
with st.expander("🎯 Step 4: Feature Selection"):
    features = [c for c in processed_df.columns if c != target_col]

    selected_features = st.multiselect("Select Features", features, default=features)

    X = processed_df[selected_features]
    y = processed_df[target_col]

# ------------------------------
# ✂️ Step 5: Train-Test Split
# ------------------------------
with st.expander("✂️ Step 5: Data Split"):
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# ------------------------------
# 🤖 Step 6: Model Selection
# ------------------------------
with st.expander("🤖 Step 6: Model Selection"):
    model_type = st.selectbox("Choose Model", ["Linear Regression", "Random Forest", "XGBoost"])

    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100)
    else:
        model = XGBRegressor(n_estimators=100)

# ------------------------------
# 🚀 Step 7: Train Model
# ------------------------------
with st.expander("🚀 Step 7: Train & Evaluate"):
    if st.button("Train Model"):

        model.fit(X_train, y_train)
        st.session_state.model = model

        y_pred = model.predict(X_test)

        st.success("Model trained successfully!")

        st.write("### 📊 Performance")
        st.write("MAE:", metrics.mean_absolute_error(y_test, y_pred))
        st.write("R² Score:", metrics.r2_score(y_test, y_pred))

# ------------------------------
# 🔮 Step 8: Prediction
# ------------------------------
with st.expander("🔮 Step 8: Predict Sales"):
    if st.session_state.model:

        inputs = {}
        for f in selected_features:
            inputs[f] = st.number_input(f, value=float(df[f].mean()))

        if st.button("Predict"):
            input_df = pd.DataFrame([inputs])

            if st.session_state.scaler:
                input_df[selected_features] = st.session_state.scaler.transform(input_df[selected_features])

            pred = st.session_state.model.predict(input_df)[0]

            st.success(f"📢 Predicted Sales: {pred:.2f}")
```
