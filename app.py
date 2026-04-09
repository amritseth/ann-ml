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
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Advertising Sales Pipeline", layout="wide")

st.title("📊 Advertising Sales Prediction Pipeline")
st.markdown("""
This dashboard implements a full Data Science pipeline to analyze the impact of advertising expenditure on product sales.
Upload your dataset and follow the steps below to build, train, and evaluate your machine learning models.
""")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'model' not in st.session_state:
    st.session_state.model = None

# --- Step 1: Input Data ---
with st.expander("📂 1. Input Data", expanded=True):
    # Check for local dataset as a default option
    local_dataset_path = "dataset_files/Advertising Budget and Sales.csv"
    import os
    
    upload_col, select_col = st.columns([2, 1])
    
    with upload_col:
        uploaded_file = st.file_uploader("Upload your Advertising Sales CSV file", type=["csv"])
    
    with select_col:
        use_sample = st.checkbox("Use extracted sample dataset", value=os.path.exists(local_dataset_path))

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Data uploaded successfully!")
    elif use_sample and os.path.exists(local_dataset_path):
        df = pd.read_csv(local_dataset_path)
        st.success("Loaded extracted sample dataset!")
    else:
        df = None

    if df is not None:
        # Drop unnamed columns if they exist
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Identify Target Column
        target_candidates = ['Sales', 'Sales ($)', 'sales']
        detected_target = next((col for col in target_candidates if col in df.columns), df.columns[-1])
        
        st.session_state.target_col = st.selectbox("Select Target Column (Sales)", df.columns, 
                                                   index=df.columns.get_loc(detected_target))
        
        st.session_state.df = df
        st.write("### Data Preview")
        st.dataframe(df.head())
        st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    else:
        st.info("Please upload a CSV file or check the sample dataset to begin.")

# Proceed only if data is loaded
if st.session_state.df is not None:
    df = st.session_state.df
    target_col = st.session_state.target_col

    # --- Step 2: Exploratory Data Analysis (EDA) ---
    with st.expander("🔍 2. Exploratory Data Analysis (EDA)"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Statistical Summary")
            st.write(df.describe())
            
        with col2:
            st.write("### Correlation Heatmap")
            corr = df.corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Feature Correlations")
            st.plotly_chart(fig_corr, use_container_width=True)
            
        st.write(f"### Visualizing Relationships ({target_col} vs. Budget)")
        features = [col for col in df.columns if col != target_col]
        selected_viz_feature = st.selectbox("Select Feature to Visualize against Target", features)
        
        fig_scatter = px.scatter(df, x=selected_viz_feature, y=target_col, trendline="ols", 
                                 title=f"{selected_viz_feature} vs {target_col}")
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.write("### Feature Distributions")
        selected_dist_feature = st.selectbox("Select Feature for Distribution", df.columns)
        fig_hist = px.histogram(df, x=selected_dist_feature, marginal="box", title=f"Distribution of {selected_dist_feature}")
        st.plotly_chart(fig_hist, use_container_width=True)

    # --- Step 3: Data Engineering & Cleaning ---
    with st.expander("🛠️ 3. Data Engineering & Cleaning"):
        st.write("### Data Integrity Check")
        missing_values = df.isnull().sum()
        if missing_values.sum() == 0:
            st.success("No missing values detected!")
        else:
            st.warning(f"Detected {missing_values.sum()} missing values.")
            if st.button("Fill Missing Values with Mean"):
                df = df.fillna(df.mean())
                st.session_state.df = df
                st.success("Missing values filled with mean.")

        st.write("### Feature Scaling")
        scaling_method = st.radio("Select Scaling Method", ["None", "StandardScaler", "MinMaxScaler"])
        
        processed_df = df.copy()
        if scaling_method != "None":
            scaler = StandardScaler() if scaling_method == "StandardScaler" else MinMaxScaler()
            cols_to_scale = [col for col in df.columns if col != target_col]
            processed_df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
            st.session_state.scaler = scaler
            st.write(f"Features scaled using {scaling_method}")
            st.dataframe(processed_df.head())
        else:
            st.session_state.scaler = None
        
        st.session_state.processed_df = processed_df

    # --- Step 4: Feature Selection ---
    with st.expander("🎯 4. Feature Selection"):
        all_features = [col for col in st.session_state.processed_df.columns if col != target_col]
        selected_features = st.multiselect("Select Features for Prediction", all_features, default=all_features)
        
        if not selected_features:
            st.error("Please select at least one feature.")
        else:
            X = st.session_state.processed_df[selected_features]
            y = st.session_state.processed_df[target_col]
            st.write(f"Selected Features: {', '.join(selected_features)}")

    # --- Step 5: Data Split ---
    with st.expander("✂️ 5. Data Split"):
        test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
        random_state = st.number_input("Random State", value=42)
        
        if selected_features:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            st.write(f"Training set size: {X_train.shape[0]} samples")
            st.write(f"Testing set size: {X_test.shape[0]} samples")

    # --- Step 6: Model Selection ---
    with st.expander("🤖 6. Model Selection"):
        model_type = st.selectbox("Choose a Model", ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor"])
        
        model_params = {}
        if model_type == "Random Forest Regressor":
            model_params['n_estimators'] = st.slider("Number of Estimators", 10, 500, 100)
            model_params['max_depth'] = st.slider("Max Depth", 1, 20, 10)
        elif model_type == "XGBoost Regressor":
            model_params['n_estimators'] = st.slider("Number of Estimators", 10, 500, 100)
            model_params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1)

    # --- Step 7 & 8: Model Training & K-Fold Validation ---
    with st.expander("🚀 7 & 8. Model Training & K-Fold Validation"):
        k_folds = st.slider("Number of Folds (K)", 2, 10, 5)
        
        if st.button("Train Model & Validate"):
            if model_type == "Linear Regression":
                model = LinearRegression()
            elif model_type == "Random Forest Regressor":
                model = RandomForestRegressor(**model_params, random_state=random_state)
            elif model_type == "XGBoost Regressor":
                model = XGBRegressor(**model_params, random_state=random_state)
            
            # Training
            model.fit(X_train, y_train)
            st.session_state.model = model
            
            # K-Fold Validation
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
            
            st.success(f"{model_type} trained successfully!")
            st.write(f"### {k_folds}-Fold Cross-Validation (R² Scores)")
            st.write(cv_scores)
            st.write(f"**Mean R² Score:** {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

    # --- Step 9: Performance Metrics ---
    with st.expander("📈 9. Performance Metrics"):
        if st.session_state.model is not None:
            model = st.session_state.model
            y_pred = model.predict(X_test)
            
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(y_test, y_pred)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE", f"{mae:.4f}")
            m2.metric("MSE", f"{mse:.4f}")
            m3.metric("RMSE", f"{rmse:.4f}")
            m4.metric("R² Score", f"{r2:.4f}")
            
            st.write("### Actual vs. Predicted")
            fig_results = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Sales', 'y': 'Predicted Sales'},
                                    title="Actual vs. Predicted Sales")
            fig_results.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                                 line=dict(color="Red", dash="dash"))
            st.plotly_chart(fig_results, use_container_width=True)
            
            # Feature Importance for tree-based models
            if model_type != "Linear Regression":
                st.write("### Feature Importance")
                importance = model.feature_importances_
                feat_importance = pd.DataFrame({'Feature': selected_features, 'Importance': importance}).sort_values(by='Importance', ascending=False)
                fig_imp = px.bar(feat_importance, x='Importance', y='Feature', orientation='h', title="Feature Importance")
                st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Train the model in the previous step to see performance metrics.")

    # --- Step 10: Real-time Prediction ---
    with st.expander("🔮 10. Real-time Prediction"):
        if st.session_state.model is not None:
            st.write("### Predict Sales for New Data")
            input_data = {}
            cols = st.columns(len(selected_features))
            
            for i, feature in enumerate(selected_features):
                with cols[i]:
                    input_data[feature] = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))
            
            if st.button("Predict Sales"):
                input_df = pd.DataFrame([input_data])
                
                # Apply scaling if it was used during training
                if st.session_state.get('scaler') is not None:
                    # Only scale features that were selected and scaled
                    input_df[selected_features] = st.session_state.scaler.transform(input_df[selected_features])
                
                prediction = st.session_state.model.predict(input_df)[0]
                st.success(f"### Predicted {target_col}: {prediction:.2f}")
                
                # Gauge chart for prediction
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction,
                    title = {'text': f"Predicted {target_col}"},
                    gauge = {'axis': {'range': [None, df[target_col].max() * 1.2]},
                             'bar': {'color': "darkblue"}}
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.info("Train the model to enable the prediction tool.")
