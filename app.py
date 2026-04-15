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

st.set_page_config(
    page_title="Sales Prediction Dashboard | Amrit Seth", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Sales Prediction Dashboard\nDeveloped by Amrit Seth & Akhilesh Yadav"
    }
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom card styling */
    .css-1r6slb0 {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #667eea;
        font-weight: bold;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        font-weight: 600;
        font-size: 18px;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2d3748;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <div style='text-align: center; padding: 20px; background: rgba(255,255,255,0.95); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 16px rgba(0,0,0,0.1);'>
        <h1 style='color: #667eea; font-size: 48px; margin-bottom: 10px;'>📊 Advertising Sales Prediction Dashboard</h1>
        <p style='font-size: 18px; color: #4a5568; margin-top: 10px;'>
            👨‍💻 <strong>Developed by Amrit Seth & Akhilesh Yadav</strong>
        </p>
        <p style='font-size: 16px; color: #718096; max-width: 800px; margin: 20px auto;'>
            Transform advertising data into actionable insights with advanced machine learning models. 
            Analyze, predict, and optimize your sales strategy with cutting-edge AI technology.
        </p>
    </div>
    """, unsafe_allow_html=True)

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

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h2 style='color: white;'>🎯 Navigation</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <div style='color: white; padding: 10px;'>
            <h4>📋 Steps:</h4>
            <ol>
                <li>Load Dataset</li>
                <li>Explore Data</li>
                <li>Clean & Scale</li>
                <li>Select Features</li>
                <li>Split Data</li>
                <li>Choose Model</li>
                <li>Train & Evaluate</li>
                <li>Make Predictions</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <div style='color: white; padding: 10px; text-align: center;'>
            <p><strong>💡 Tip:</strong> Follow the steps in order for best results!</p>
        </div>
    """, unsafe_allow_html=True)

# ------------------------------
# 📂 Step 1: Load Dataset
# ------------------------------

with st.expander("📂 **STEP 1: Load Dataset**", expanded=True):
    
    st.markdown("""
        <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='color: white; margin: 0;'>📥 Upload Your Data</h3>
            <p style='color: white; opacity: 0.9; margin: 5px 0 0 0;'>
                Start by uploading your CSV file or use our sample dataset
            </p>
        </div>
    """, unsafe_allow_html=True)

    local_dataset_path = "dataset_files/Advertising Budget and Sales.csv"

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader("📎 Choose a CSV file", type=["csv"])

    with col2:
        use_sample = st.checkbox("✨ Use Sample Dataset", value=os.path.exists(local_dataset_path))

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
            "🎯 Select Target Column (What do you want to predict?)",
            df.columns,
            index=df.columns.get_loc(detected_target)
        )

        st.session_state.df = df

        st.markdown("### 🔍 Data Preview")
        
        # Display data in a nice container
        st.dataframe(df.head(10), use_container_width=True)
        
        # Stats in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Rows", df.shape[0])
        with col2:
            st.metric("📋 Total Columns", df.shape[1])
        with col3:
            st.metric("🎯 Target Variable", st.session_state.target_col)

    else:
        st.info("📁 Please upload a dataset to begin your analysis journey")

# ------------------------------
# Continue if data exists
# ------------------------------

if st.session_state.df is not None:

    df = st.session_state.df
    target_col = st.session_state.target_col

    # ------------------------------
    # 🔍 Step 2: EDA
    # ------------------------------
    with st.expander("🔍 **STEP 2: Exploratory Data Analysis**"):
        
        st.markdown("""
            <div style='background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%); 
                        padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
                <h3 style='color: white; margin: 0;'>📊 Discover Data Insights</h3>
                <p style='color: white; opacity: 0.9; margin: 5px 0 0 0;'>
                    Explore statistical summaries and relationships in your data
                </p>
            </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📈 Statistics", "🔥 Correlations", "📊 Visualizations"])
        
        with tab1:
            st.markdown("#### 📊 Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)

        with tab2:
            st.markdown("#### 🔥 Correlation Heatmap")
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                fig_corr = px.imshow(
                    numeric_df.corr(), 
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    aspect='auto'
                )
                fig_corr.update_layout(
                    title="Feature Correlation Matrix",
                    height=600
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("No numeric columns for correlation.")

        with tab3:
            features = [col for col in df.columns if col != target_col and df[col].dtype in [np.float64, np.int64]]

            if features:
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    selected_feature = st.selectbox("📈 Choose Feature", features)
                
                with col2:
                    fig_scatter = px.scatter(
                        df, 
                        x=selected_feature, 
                        y=target_col, 
                        trendline="ols",
                        title=f"📈 {selected_feature} vs {target_col}",
                        color_discrete_sequence=['#667eea']
                    )
                    fig_scatter.update_layout(height=500)
                    st.plotly_chart(fig_scatter, use_container_width=True)

    # ------------------------------
    # 🛠 Step 3: Data Cleaning & Scaling
    # ------------------------------
    with st.expander("🛠 **STEP 3: Data Cleaning & Scaling**"):
        
        st.markdown("""
            <div style='background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
                <h3 style='color: white; margin: 0;'>🧹 Prepare Your Data</h3>
                <p style='color: white; opacity: 0.9; margin: 5px 0 0 0;'>
                    Clean and normalize your data for optimal model performance
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Check for missing values
        missing_count = df.isnull().sum().sum()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if missing_count > 0:
                st.warning(f"⚠️ Found {missing_count} missing values")
                if st.button("🔧 Fill Missing Values"):
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                    st.session_state.df = df
                    st.success("✅ Missing values filled with mean!")
                    st.rerun()
            else:
                st.success("✅ No missing values found! Data is clean.")
        
        with col2:
            st.metric("Missing Values", missing_count, delta=None)

        st.markdown("---")
        
        st.markdown("#### ⚙️ Feature Scaling")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            scaling = st.radio(
                "Choose Scaling Method",
                ["None", "StandardScaler", "MinMaxScaler"],
                help="StandardScaler: Mean=0, Std=1 | MinMaxScaler: Range [0,1]"
            )

        processed_df = df.copy()

        if scaling != "None":
            scaler = StandardScaler() if scaling == "StandardScaler" else MinMaxScaler()
            cols_to_scale = [c for c in df.columns if c != target_col and df[c].dtype in [np.float64, np.int64]]

            if cols_to_scale:
                processed_df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
                st.session_state.scaler = scaler
                st.success(f"✅ Applied {scaling} to {len(cols_to_scale)} features")
                
                # Show comparison
                with col2:
                    st.info(f"📊 Scaled {len(cols_to_scale)} features using {scaling}")
            else:
                st.warning("No numeric features to scale")
                st.session_state.scaler = None
        else:
            st.session_state.scaler = None

        st.session_state.processed_df = processed_df

        # Show comparison if scaling applied
        if scaling != "None" and cols_to_scale:
            st.markdown("#### 📊 Before and After Scaling")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Data**")
                st.dataframe(df[cols_to_scale].head(), use_container_width=True)
            with col2:
                st.markdown("**Scaled Data**")
                st.dataframe(processed_df[cols_to_scale].head(), use_container_width=True)

    # ------------------------------
    # 🎯 Step 4: Feature Selection
    # ------------------------------
    with st.expander("🎯 **STEP 4: Feature Selection**"):
        
        st.markdown("""
            <div style='background: linear-gradient(90deg, #fa709a 0%, #fee140 100%); 
                        padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
                <h3 style='color: white; margin: 0;'>🎯 Choose Your Features</h3>
                <p style='color: white; opacity: 0.9; margin: 5px 0 0 0;'>
                    Select which features to use for training your model
                </p>
            </div>
        """, unsafe_allow_html=True)

        all_features = [c for c in st.session_state.processed_df.columns 
                        if c != target_col and st.session_state.processed_df[c].dtype in [np.float64, np.int64]]

        if not all_features:
            st.error("❌ No numeric features available for modeling")
            st.stop()

        selected_features = st.multiselect(
            "✨ Select Features for Training",
            all_features,
            default=all_features,
            help="Choose one or more features to train your model"
        )

        if not selected_features:
            st.warning("⚠️ Please select at least one feature to continue.")
            st.stop()

        st.session_state.selected_features = selected_features

        X = st.session_state.processed_df[selected_features]
        y = st.session_state.processed_df[target_col]

        # Display selected features nicely
        st.success(f"✅ Selected {len(selected_features)} features for training")
        
        cols = st.columns(min(4, len(selected_features)))
        for idx, feature in enumerate(selected_features):
            with cols[idx % len(cols)]:
                st.metric(f"Feature {idx+1}", feature)

    # ------------------------------
    # ✂️ Step 5: Train-Test Split
    # ------------------------------
    with st.expander("✂️ **STEP 5: Data Split**"):
        
        st.markdown("""
            <div style='background: linear-gradient(90deg, #30cfd0 0%, #330867 100%); 
                        padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
                <h3 style='color: white; margin: 0;'>✂️ Split Your Data</h3>
                <p style='color: white; opacity: 0.9; margin: 5px 0 0 0;'>
                    Divide data into training and testing sets
                </p>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("🎚️ Test Size (%)", 10, 50, 20, help="Percentage of data to use for testing") / 100
        
        with col2:
            random_state = st.number_input("🎲 Random State", value=42, min_value=0, help="Seed for reproducibility")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )

        # Store in session state
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        # Display split results
        col1, col2, col3 = st.columns(3)
        col1.metric("🏋️ Training Samples", X_train.shape[0], delta=f"{(1-test_size)*100:.0f}%")
        col2.metric("🧪 Testing Samples", X_test.shape[0], delta=f"{test_size*100:.0f}%")
        col3.metric("📊 Total Features", X_train.shape[1])
        
        # Visualization of split
        fig_split = go.Figure(data=[
            go.Pie(
                labels=['Training Set', 'Testing Set'],
                values=[len(X_train), len(X_test)],
                hole=.4,
                marker_colors=['#667eea', '#764ba2']
            )
        ])
        fig_split.update_layout(
            title_text="Data Split Distribution",
            height=400
        )
        st.plotly_chart(fig_split, use_container_width=True)

    # ------------------------------
    # 🤖 Step 6: Model Selection
    # ------------------------------
    with st.expander("🤖 **STEP 6: Model Selection**"):
        
        st.markdown("""
            <div style='background: linear-gradient(90deg, #a8edea 0%, #fed6e3 100%); 
                        padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
                <h3 style='color: #2d3748; margin: 0;'>🤖 Choose Your AI Model</h3>
                <p style='color: #4a5568; margin: 5px 0 0 0;'>
                    Select and configure your machine learning algorithm
                </p>
            </div>
        """, unsafe_allow_html=True)

        model_type = st.selectbox(
            "🎯 Choose Model Algorithm",
            ["Linear Regression", "Random Forest", "XGBoost"],
            help="Different algorithms have different strengths"
        )

        if model_type == "Linear Regression":
            st.info("📐 **Linear Regression**: Simple and interpretable, best for linear relationships")
            model = LinearRegression()
            
            st.markdown("""
                **Pros:**
                - Fast training
                - Easy to interpret
                - Works well with linear data
                
                **Best for:** Simple linear relationships
            """)

        elif model_type == "Random Forest":
            st.info("🌲 **Random Forest**: Ensemble of decision trees, robust and accurate")
            
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("🌲 Number of Trees", 10, 300, 100, help="More trees = better performance but slower")
            with col2:
                max_depth = st.slider("📏 Max Depth", 1, 20, 10, help="Maximum depth of each tree")
            
            model = RandomForestRegressor(
                n_estimators=n_estimators, 
                max_depth=max_depth,
                random_state=int(random_state)
            )
            
            st.markdown(f"""
                **Configuration:**
                - Trees: {n_estimators}
                - Max Depth: {max_depth}
                
                **Best for:** Non-linear relationships, feature importance analysis
            """)

        else:  # XGBoost
            st.info("⚡ **XGBoost**: Gradient boosting, state-of-the-art performance")
            
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("🔢 Number of Estimators", 10, 300, 100)
            with col2:
                learning_rate = st.slider("📈 Learning Rate", 0.01, 0.3, 0.1, help="Lower = more accurate but slower")
            
            model = XGBRegressor(
                n_estimators=n_estimators, 
                learning_rate=learning_rate, 
                random_state=int(random_state)
            )
            
            st.markdown(f"""
                **Configuration:**
                - Estimators: {n_estimators}
                - Learning Rate: {learning_rate}
                
                **Best for:** Complex patterns, competition-grade performance
            """)

        st.session_state.model_config = {
            'type': model_type,
            'model': model
        }

    # ------------------------------
    # 🚀 Step 7: Train & Validate
    # ------------------------------
    with st.expander("🚀 **STEP 7: Train & Evaluate**"):
        
        st.markdown("""
            <div style='background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 100%); 
                        padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
                <h3 style='color: #2d3748; margin: 0;'>🚀 Train Your Model</h3>
                <p style='color: #4a5568; margin: 5px 0 0 0;'>
                    Train and evaluate your machine learning model
                </p>
            </div>
        """, unsafe_allow_html=True)

        if st.button("🎯 **Train Model**", use_container_width=True):

            with st.spinner("🔄 Training model... Please wait..."):
                
                # Progress bar
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
                
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

                st.balloons()
                st.success("✅ Model trained successfully!")

                # Display metrics in beautiful cards
                st.markdown("### 📊 Model Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 20px; border-radius: 15px; text-align: center; color: white;'>
                            <h4 style='margin: 0; color: white;'>MAE</h4>
                            <h2 style='margin: 10px 0; color: white;'>{mae:.4f}</h2>
                            <p style='margin: 0; opacity: 0.8; font-size: 12px;'>Mean Absolute Error</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                    padding: 20px; border-radius: 15px; text-align: center; color: white;'>
                            <h4 style='margin: 0; color: white;'>RMSE</h4>
                            <h2 style='margin: 10px 0; color: white;'>{rmse:.4f}</h2>
                            <p style='margin: 0; opacity: 0.8; font-size: 12px;'>Root Mean Squared Error</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    r2_color = "#10b981" if r2 > 0.7 else "#f59e0b" if r2 > 0.5 else "#ef4444"
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                                    padding: 20px; border-radius: 15px; text-align: center; color: white;'>
                            <h4 style='margin: 0; color: white;'>R² (Test)</h4>
                            <h2 style='margin: 10px 0; color: white;'>{r2:.4f}</h2>
                            <p style='margin: 0; opacity: 0.8; font-size: 12px;'>Test Accuracy</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                                    padding: 20px; border-radius: 15px; text-align: center; color: white;'>
                            <h4 style='margin: 0; color: white;'>R² (Train)</h4>
                            <h2 style='margin: 10px 0; color: white;'>{train_r2:.4f}</h2>
                            <p style='margin: 0; opacity: 0.8; font-size: 12px;'>Training Accuracy</p>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # Prediction vs Actual plot
                st.markdown("### 📈 Predictions vs Actual Values")
                
                fig = go.Figure()
                
                # Actual vs Predicted scatter
                fig.add_trace(go.Scatter(
                    x=y_test,
                    y=y_pred_test,
                    mode='markers',
                    name='Predictions',
                    marker=dict(
                        size=10, 
                        color=y_pred_test,
                        colorscale='Viridis',
                        showscale=True,
                        line=dict(width=1, color='white')
                    )
                ))
                
                # Perfect prediction line
                min_val = min(y_test.min(), y_pred_test.min())
                max_val = max(y_test.max(), y_pred_test.max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash', width=3)
                ))
                
                fig.update_layout(
                    xaxis_title='Actual Values',
                    yaxis_title='Predicted Values',
                    title='Actual vs Predicted Values',
                    height=500,
                    hovermode='closest',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)

                # Residual plot
                residuals = y_test - y_pred_test
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_residual = go.Figure()
                    fig_residual.add_trace(go.Scatter(
                        x=y_pred_test,
                        y=residuals,
                        mode='markers',
                        marker=dict(size=8, color='#667eea', opacity=0.6)
                    ))
                    fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
                    fig_residual.update_layout(
                        title="Residual Plot",
                        xaxis_title="Predicted Values",
                        yaxis_title="Residuals",
                        height=400
                    )
                    st.plotly_chart(fig_residual, use_container_width=True)
                
                with col2:
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=residuals,
                        marker_color='#764ba2',
                        nbinsx=30
                    ))
                    fig_hist.update_layout(
                        title="Residual Distribution",
                        xaxis_title="Residuals",
                        yaxis_title="Frequency",
                        height=400
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                # Feature importance (for tree-based models)
                if model_type in ["Random Forest", "XGBoost"]:
                    st.markdown("### 🎯 Feature Importance Analysis")
                    
                    feature_importance = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig_imp = px.bar(
                        feature_importance, 
                        x='Importance', 
                        y='Feature', 
                        orientation='h',
                        color='Importance',
                        color_continuous_scale='Viridis',
                        title='Feature Importance Ranking'
                    )
                    fig_imp.update_layout(height=400)
                    st.plotly_chart(fig_imp, use_container_width=True)

    # ------------------------------
    # 🔮 Step 8: Prediction
    # ------------------------------
    with st.expander("🔮 **STEP 8: Make Predictions**"):
        
        st.markdown("""
            <div style='background: linear-gradient(90deg, #ffecd2 0%, #fcb69f 100%); 
                        padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
                <h3 style='color: #2d3748; margin: 0;'>🔮 Predict Sales</h3>
                <p style='color: #4a5568; margin: 5px 0 0 0;'>
                    Use your trained model to predict sales for new data
                </p>
            </div>
        """, unsafe_allow_html=True)

        if st.session_state.model is not None:

            st.markdown("### 📝 Enter Feature Values")
            
            selected_features = st.session_state.selected_features
            input_data = {}

            # Create input fields in a nice grid
            num_cols = min(3, len(selected_features))
            cols = st.columns(num_cols)
            
            for idx, feature in enumerate(selected_features):
                with cols[idx % num_cols]:
                    input_data[feature] = st.number_input(
                        f"💰 {feature}",
                        value=float(df[feature].mean()),
                        format="%.2f",
                        help=f"Average: {df[feature].mean():.2f}"
                    )

            st.markdown("---")
            
            if st.button("🔮 **Predict Sales**", use_container_width=True):

                input_df = pd.DataFrame([input_data])

                # Apply scaling if used
                if st.session_state.scaler is not None:
                    input_df[selected_features] = st.session_state.scaler.transform(input_df[selected_features])

                # Make prediction
                prediction = st.session_state.model.predict(input_df)[0]

                # Animated success message
                st.balloons()
                
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 30px; border-radius: 20px; text-align: center; margin: 20px 0;
                                box-shadow: 0 10px 25px rgba(0,0,0,0.2);'>
                        <h2 style='color: white; margin: 0;'>🎉 Prediction Result</h2>
                        <h1 style='color: white; font-size: 48px; margin: 20px 0;'>${prediction:,.2f}</h1>
                        <p style='color: white; opacity: 0.9; font-size: 18px;'>Predicted {target_col}</p>
                    </div>
                """, unsafe_allow_html=True)

                # Gauge chart
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prediction,
                        title={'text': f"Predicted {target_col}", 'font': {'size': 24}},
                        delta={'reference': df[target_col].mean()},
                        gauge={
                            'axis': {'range': [None, df[target_col].max() * 1.2]},
                            'bar': {'color': "#667eea"},
                            'steps': [
                                {'range': [0, df[target_col].quantile(0.33)], 'color': "#fecfef"},
                                {'range': [df[target_col].quantile(0.33), df[target_col].quantile(0.66)], 'color': "#fad0c4"},
                                {'range': [df[target_col].quantile(0.66), df[target_col].max() * 1.2], 'color': "#a1c4fd"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': df[target_col].mean()
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=400)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col2:
                    st.markdown("#### 📊 Comparison")
                    st.metric("Prediction", f"${prediction:,.2f}")
                    st.metric("Dataset Average", f"${df[target_col].mean():,.2f}", 
                             delta=f"{((prediction/df[target_col].mean() - 1) * 100):.1f}%")
                    st.metric("Dataset Max", f"${df[target_col].max():,.2f}")
                    
                    if prediction > df[target_col].quantile(0.75):
                        st.success("🎯 High predicted value!")
                    elif prediction < df[target_col].quantile(0.25):
                        st.warning("⚠️ Low predicted value")
                    else:
                        st.info("📊 Average predicted value")

        else:
            st.warning("⚠️ Please train the model first in Step 7 before making predictions.")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 30px; background: rgba(255,255,255,0.95); 
                border-radius: 15px; margin-top: 30px;'>
        <h3 style='color: #667eea; margin-bottom: 10px;'>🚀 Sales Prediction Dashboard</h3>
        <p style='color: #4a5568; font-size: 16px;'>
            <strong>👨‍💻 Developed by Amrit Seth & Akhilesh Yadav</strong>
        </p>
        <p style='color: #718096; font-size: 14px; margin-top: 10px;'>
            Machine Learning Pipeline Demo | Built with Streamlit, scikit-learn & XGBoost
        </p>
        <div style='margin-top: 20px;'>
            <span style='margin: 0 10px; color: #667eea;'>⭐ Star on GitHub</span>
            <span style='margin: 0 10px; color: #667eea;'>📧 Contact</span>
            <span style='margin: 0 10px; color: #667eea;'>📚 Documentation</span>
        </div>
    </div>
""", unsafe_allow_html=True)
