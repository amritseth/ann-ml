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
👨‍💻 **Developed by Amrit Seth & Akhilesh Yadav**

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

if 'target_col' not in st.session_state:
    st.session_state.target_col = None

if 'selected_features' not in st.session_state:
    st.session_state.selected_features = None

if 'X_train' not in st.session_state:
    st.session_state.X_train = None

if 'X_test' not in st.session_state:
    st.session_state.X_test = None

if 'y_train' not in st.session_state:
    st.session_state.y_train = None

if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# ------------------------------
# 📂 Step 1: Load Dataset
# ------------------------------

with st.expander("📂 Step 1: Load Dataset", expanded=True):

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
        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Detect target column
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
        st.info("📁 Upload a dataset to begin.")

# ------------------------------
# Continue if data exists
# ------------------------------

if st.session_state.df is not None:

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
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                fig_corr = px.imshow(numeric_df.corr(), text_auto=True, color_continuous_scale='RdBu_r')
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("No numeric columns for correlation.")

        features = [col for col in df.columns if col != target_col and df[col].dtype in [np.float64, np.int64]]

        if features:
            selected_feature = st.selectbox("📈 Feature vs Target", features)

            fig_scatter = px.scatter(df, x=selected_feature, y=target_col, trendline="ols",
                                     title=f"{selected_feature} vs {target_col}")
            st.plotly_chart(fig_scatter, use_container_width=True)

    # ------------------------------
    # 🛠 Step 3: Data Cleaning & Scaling
    # ------------------------------
    with st.expander("🛠 Step 3: Data Cleaning & Scaling"):

        # Check for missing values
        missing_count = df.isnull().sum().sum()
        
        if missing_count > 0:
            st.warning(f"⚠️ Found {missing_count} missing values")
            if st.button("Fill Missing Values"):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                st.session_state.df = df
                st.success("✅ Missing values filled!")
                st.rerun()
        else:
            st.success("✅ No missing values found!")

        scaling = st.radio("Scaling Method", ["None", "StandardScaler", "MinMaxScaler"])

        processed_df = df.copy()

        if scaling != "None":
            scaler = StandardScaler() if scaling == "StandardScaler" else MinMaxScaler()
            cols_to_scale = [c for c in df.columns if c != target_col and df[c].dtype in [np.float64, np.int64]]

            if cols_to_scale:
                processed_df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
                st.session_state.scaler = scaler
                st.success(f"✅ Applied {scaling} to {len(cols_to_scale)} features")
            else:
                st.warning("No numeric features to scale")
                st.session_state.scaler = None

        else:
            st.session_state.scaler = None

        st.session_state.processed_df = processed_df

        # Show comparison
        if scaling != "None" and cols_to_scale:
            st.write("### Before and After Scaling")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Data**")
                st.dataframe(df[cols_to_scale].head())
            with col2:
                st.write("**Scaled Data**")
                st.dataframe(processed_df[cols_to_scale].head())

    # ------------------------------
    # 🎯 Step 4: Feature Selection
    # ------------------------------
    with st.expander("🎯 Step 4: Feature Selection"):

        all_features = [c for c in st.session_state.processed_df.columns 
                        if c != target_col and st.session_state.processed_df[c].dtype in [np.float64, np.int64]]

        if not all_features:
            st.error("❌ No numeric features available for modeling")
            st.stop()

        selected_features = st.multiselect(
            "Select Features for Training",
            all_features,
            default=all_features
        )

        if not selected_features:
            st.warning("⚠️ Please select at least one feature.")
            st.stop()

        st.session_state.selected_features = selected_features

        X = st.session_state.processed_df[selected_features]
        y = st.session_state.processed_df[target_col]

        st.success(f"✅ Selected {len(selected_features)} features")

    # ------------------------------
    # ✂️ Step 5: Train-Test Split
    # ------------------------------
    with st.expander("✂️ Step 5: Data Split"):

        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
        random_state = st.number_input("Random State", value=42, min_value=0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )

        # Store in session state
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        col1, col2 = st.columns(2)
        col1.metric("🏋️ Training Samples", X_train.shape[0])
        col2.metric("🧪 Testing Samples", X_test.shape[0])

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
            st.info("Simple linear regression model selected")

        elif model_type == "Random Forest":
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("Number of Trees", 10, 300, 100)
            with col2:
                max_depth = st.slider("Max Depth", 1, 20, 10)
            
            model = RandomForestRegressor(
                n_estimators=n_estimators, 
                max_depth=max_depth,
                random_state=int(random_state)
            )

        else:  # XGBoost
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("Number of Estimators", 10, 300, 100)
            with col2:
                learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
            
            model = XGBRegressor(
                n_estimators=n_estimators, 
                learning_rate=learning_rate, 
                random_state=int(random_state)
            )

        st.session_state.model_config = {
            'type': model_type,
            'model': model
        }

    # ------------------------------
    # 🚀 Step 7: Train & Validate
    # ------------------------------
    with st.expander("🚀 Step 7: Train & Evaluate"):

        if st.button("🎯 Train Model"):

            with st.spinner("Training model..."):
                
                # Get data from session state
                X_train = st.session_state.X_train
                X_test = st.session_state.X_test
                y_train = st.session_state.y_train
                y_test = st.session_state.y_test
                
                model = st.session_state.model_config['model']
                
                # Train model
                model.fit(X_train, y_train)
                st.session_state.model = model

                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                # Calculate metrics
                mae = metrics.mean_absolute_error(y_test, y_pred_test)
                mse = metrics.mean_squared_error(y_test, y_pred_test)
                rmse = np.sqrt(mse)
                r2 = metrics.r2_score(y_test, y_pred_test)
                
                # Training metrics
                train_r2 = metrics.r2_score(y_train, y_pred_train)

                st.success("✅ Model trained successfully!")

                # Display metrics
                st.write("### 📊 Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", f"{mae:.4f}")
                col2.metric("RMSE", f"{rmse:.4f}")
                col3.metric("R² (Test)", f"{r2:.4f}")
                col4.metric("R² (Train)", f"{train_r2:.4f}")

                # Prediction vs Actual plot
                st.write("### 📈 Predictions vs Actual Values")
                
                fig = go.Figure()
                
                # Actual vs Predicted scatter
                fig.add_trace(go.Scatter(
                    x=y_test,
                    y=y_pred_test,
                    mode='markers',
                    name='Predictions',
                    marker=dict(size=8, color='blue', opacity=0.6)
                ))
                
                # Perfect prediction line
                min_val = min(y_test.min(), y_pred_test.min())
                max_val = max(y_test.max(), y_pred_test.max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    xaxis_title='Actual Values',
                    yaxis_title='Predicted Values',
                    title='Actual vs Predicted'
                )
                
                st.plotly_chart(fig, use_container_width=True)

                # Feature importance (for tree-based models)
                if model_type in ["Random Forest", "XGBoost"]:
                    st.write("### 🎯 Feature Importance")
                    
                    feature_importance = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig_imp = px.bar(feature_importance, x='Importance', y='Feature', orientation='h')
                    st.plotly_chart(fig_imp, use_container_width=True)

    # ------------------------------
    # 🔮 Step 8: Prediction
    # ------------------------------
    with st.expander("🔮 Step 8: Predict Sales"):

        if st.session_state.model is not None:

            st.write("### Enter Feature Values")
            
            selected_features = st.session_state.selected_features
            input_data = {}

            # Create input fields
            cols = st.columns(min(3, len(selected_features)))
            for idx, feature in enumerate(selected_features):
                with cols[idx % len(cols)]:
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        value=float(df[feature].mean()),
                        format="%.2f"
                    )

            if st.button("🔮 Predict"):

                input_df = pd.DataFrame([input_data])

                # Apply scaling if used
                if st.session_state.scaler is not None:
                    input_df[selected_features] = st.session_state.scaler.transform(input_df[selected_features])

                # Make prediction
                prediction = st.session_state.model.predict(input_df)[0]

                st.success(f"### 📢 Predicted {target_col}: **{prediction:.2f}**")

                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction,
                    title={'text': f"Predicted {target_col}"},
                    gauge={
                        'axis': {'range': [None, df[target_col].max() * 1.2]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, df[target_col].mean()], 'color': "lightgray"},
                            {'range': [df[target_col].mean(), df[target_col].max()], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': df[target_col].mean()
                        }
                    }
                ))

                st.plotly_chart(fig_gauge, use_container_width=True)

        else:
            st.info("⚠️ Please train the model first in Step 7.")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("**👨‍💻 Developed by Amrit Seth & Akhilesh Yadav** | Machine Learning Pipeline Demo")
