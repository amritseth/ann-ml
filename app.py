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
import os

# ------------------------------

# 🔧 Page Configuration

# ------------------------------

st.set_page_config(page_title="Sales Prediction Dashboard | Amrit Seth", layout="wide")

st.title("📊 Advertising Sales Prediction Dashboard")
st.markdown("""
👨‍💻 **Developed by Amrit Seth**

This project demonstrates a complete **Machine Learning pipeline** where I analyze how advertising budgets
affect product sales and build predictive models.
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

# 📂 Step 1: Load Dataset

# ------------------------------

with st.expander("📂 Step 1: Load Dataset", expanded=True):

```
local_dataset_path = "dataset_files/Advertising Budget and Sales.csv"

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

with col2:
    use_sample = st.checkbox("Use Sample Dataset", value=os.path.exists(local_dataset_path))

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")

elif use_sample and os.path.exists(local_dataset_path):
    df = pd.read_csv(local_dataset_path)
    st.success("✅ Sample dataset loaded!")

else:
    df = None

if df is not None:
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    target_candidates = ['Sales', 'sales']
    detected_target = next((c for c in target_candidates if c in df.columns), df.columns[-1])

    st.session_state.target_col = st.selectbox(
        "🎯 Select Target Column",
        df.columns,
        index=df.columns.get_loc(detected_target)
    )

    st.session_state.df = df

    st.write("### 🔍 Data Preview")
    st.dataframe(df.head())
    st.write(f"📌 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

else:
    st.info("Upload a dataset to begin.")
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
        fig_corr = px.imshow(df.corr(), text_auto=True)
        st.plotly_chart(fig_corr, use_container_width=True)

    features = [col for col in df.columns if col != target_col]

    selected_feature = st.selectbox("📈 Feature vs Target", features)

    fig_scatter = px.scatter(df, x=selected_feature, y=target_col, trendline="ols")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------------------
# 🛠 Step 3: Data Cleaning & Scaling
# ------------------------------
with st.expander("🛠 Step 3: Data Cleaning & Scaling"):

    if df.isnull().sum().sum() > 0:
        if st.button("Fill Missing Values"):
            df = df.fillna(df.mean())
            st.session_state.df = df
            st.success("Missing values filled!")

    scaling = st.radio("Scaling Method", ["None", "StandardScaler", "MinMaxScaler"])

    processed_df = df.copy()

    if scaling != "None":
        scaler = StandardScaler() if scaling == "StandardScaler" else MinMaxScaler()
        cols_to_scale = [c for c in df.columns if c != target_col]

        processed_df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        st.session_state.scaler = scaler

    else:
        st.session_state.scaler = None

    st.session_state.processed_df = processed_df

# ------------------------------
# 🎯 Step 4: Feature Selection
# ------------------------------
with st.expander("🎯 Step 4: Feature Selection"):

    all_features = [c for c in st.session_state.processed_df.columns if c != target_col]

    selected_features = st.multiselect(
        "Select Features",
        all_features,
        default=all_features
    )

    if not selected_features:
        st.warning("Please select at least one feature.")
        st.stop()

    X = st.session_state.processed_df[selected_features]
    y = st.session_state.processed_df[target_col]

# ------------------------------
# ✂️ Step 5: Train-Test Split
# ------------------------------
with st.expander("✂️ Step 5: Data Split"):

    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
    random_state = st.number_input("Random State", value=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    st.write(f"Training samples: {X_train.shape[0]}")
    st.write(f"Testing samples: {X_test.shape[0]}")

# ------------------------------
# 🤖 Step 6: Model Selection
# ------------------------------
with st.expander("🤖 Step 6: Model Selection"):

    model_type = st.selectbox(
        "Choose Model",
        ["Linear Regression", "Random Forest", "XGBoost"]
    )

    if model_type == "Linear Regression":
        model = LinearRegression()

    elif model_type == "Random Forest":
        n_estimators = st.slider("n_estimators", 10, 300, 100)
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    else:
        n_estimators = st.slider("n_estimators", 10, 300, 100)
        learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.1)
        model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)

# ------------------------------
# 🚀 Step 7: Train & Validate
# ------------------------------
with st.expander("🚀 Step 7: Train & Evaluate"):

    if st.button("Train Model"):

        model.fit(X_train, y_train)
        st.session_state.model = model

        y_pred = model.predict(X_test)

        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(y_test, y_pred)

        st.success("Model trained successfully!")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:.4f}")
        col2.metric("MSE", f"{mse:.4f}")
        col3.metric("RMSE", f"{rmse:.4f}")
        col4.metric("R²", f"{r2:.4f}")

# ------------------------------
# 🔮 Step 8: Prediction
# ------------------------------
with st.expander("🔮 Step 8: Predict Sales"):

    if st.session_state.model is not None:

        input_data = {}

        for feature in selected_features:
            input_data[feature] = st.number_input(
                f"{feature}",
                value=float(df[feature].mean())
            )

        if st.button("Predict"):

            input_df = pd.DataFrame([input_data])

            if st.session_state.scaler is not None:
                input_df[selected_features] = st.session_state.scaler.transform(input_df[selected_features])

            prediction = st.session_state.model.predict(input_df)[0]

            st.success(f"📢 Predicted {target_col}: {prediction:.2f}")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': f"Predicted {target_col}"}
            ))

            st.plotly_chart(fig_gauge, use_container_width=True)

    else:
        st.info("Train the model first.")
```
