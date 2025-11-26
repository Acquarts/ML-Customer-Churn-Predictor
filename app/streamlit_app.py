"""
Customer Churn Prediction App
=============================
Professional Streamlit application for predicting customer churn.
Features real-time predictions, batch processing, and model insights.

Author: Your Name
Date: 2024
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM STYLING
# =============================================================================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
    
    /* Main container */
    .main {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #ffffff !important;
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
        line-height: 1.3 !important;
        clear: both !important;
    }

    /* Body text */
    p, span, label, .stMarkdown {
        font-family: 'Inter', sans-serif !important;
    }

    /* Fix expander header spacing */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        padding: 0.5rem 0 !important;
    }

    /* Ensure proper spacing for sections */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e3f 0%, #2d2d5a 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    /* Risk badge styles */
    .risk-high {
        background: linear-gradient(135deg, #ff4757 0%, #ff6b6b 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 14px;
        display: inline-block;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #ffa502 0%, #ffbe0b 100%);
        color: #1a1a2e;
        padding: 8px 20px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 14px;
        display: inline-block;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #2ed573 0%, #7bed9f 100%);
        color: #1a1a2e;
        padding: 8px 20px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 14px;
        display: inline-block;
    }
    
    /* Prediction result box */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 32px;
        text-align: center;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        margin: 20px 0;
    }
    
    .prediction-box h2 {
        font-size: 64px;
        margin: 0;
        font-weight: 700;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #1a1a2e 100%);
    }
    
    section[data-testid="stSidebar"] h1 {
        color: #667eea !important;
    }
    
    /* Input fields */
    .stSelectbox, .stNumberInput {
        background-color: rgba(255,255,255,0.05);
        border-radius: 8px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 12px 24px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Feature cards */
    .feature-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_resource
def load_predictor():
    """Load the trained model and preprocessor."""
    try:
        from src.models.predict import ChurnPredictor
        return ChurnPredictor(config_path=str(project_root / "config" / "config.yaml"))
    except FileNotFoundError:
        return None


def create_gauge_chart(probability: float) -> go.Figure:
    """Create a stylish gauge chart for churn probability."""
    
    # Determine color based on probability
    if probability >= 0.7:
        color = "#ff4757"
    elif probability >= 0.4:
        color = "#ffa502"
    else:
        color = "#2ed573"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': '%', 'font': {'size': 48, 'color': 'white'}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 2,
                'tickcolor': "white",
                'tickfont': {'color': 'white'}
            },
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': 'rgba(46, 213, 115, 0.2)'},
                {'range': [40, 70], 'color': 'rgba(255, 165, 2, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(255, 71, 87, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.8,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'family': 'Space Grotesk'},
        height=280,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    """Create horizontal bar chart for feature importance."""
    
    # Take top 10
    df = importance_df.head(10).sort_values('importance')
    
    fig = go.Figure(go.Bar(
        x=df['importance'],
        y=df['feature'],
        orientation='h',
        marker=dict(
            color=df['importance'],
            colorscale=[[0, '#667eea'], [1, '#764ba2']],
            line=dict(width=0)
        ),
        text=[f'{v:.3f}' for v in df['importance']],
        textposition='outside',
        textfont=dict(color='white', size=12)
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'family': 'Inter'},
        height=400,
        margin=dict(l=20, r=60, t=20, b=20),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=11)
        )
    )
    
    return fig


def create_distribution_chart(data: pd.DataFrame, column: str) -> go.Figure:
    """Create distribution chart for a column."""
    
    fig = px.histogram(
        data, x=column,
        color_discrete_sequence=['#667eea'],
        opacity=0.8
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        showlegend=False,
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    return fig


def get_risk_badge(risk_level: str) -> str:
    """Get HTML badge for risk level."""
    risk_class = f"risk-{risk_level.lower()}"
    return f'<span class="{risk_class}">{risk_level} Risk</span>'


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        # 🔮 Customer Churn Predictor
        **AI-powered customer retention intelligence**
        """)
    
    # Load model
    predictor = load_predictor()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Navigation")
        
        page = st.radio(
            "Select Page",
            ["🎯 Single Prediction", "📊 Batch Analysis", "📈 Model Insights"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        st.markdown("""
        ### 📖 About
        
        This application uses Machine Learning to predict customer churn probability 
        based on customer attributes and behavior patterns.
        
        **Features:**
        - Real-time predictions
        - Risk factor analysis
        - Batch processing
        - Model explainability
        """)
        
        st.markdown("---")
        st.markdown("Made with ❤️ using Streamlit")
    
    # Check if model is loaded
    if predictor is None:
        st.warning("""
        ⚠️ **Model not found!**
        
        Please train the model first by running:
        ```bash
        python src/models/train.py
        ```
        
        For now, showing demo mode with simulated predictions.
        """)
        demo_mode = True
    else:
        demo_mode = False
    
    # ==========================================================================
    # PAGE: SINGLE PREDICTION
    # ==========================================================================
    if page == "🎯 Single Prediction":
        st.markdown("### Enter Customer Information")
        
        # Input form with columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 👤 Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        with col2:
            st.markdown("##### 📞 Services")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        
        with col3:
            st.markdown("##### 💳 Account")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 118.0, 50.0)
        
        # Additional services (collapsible)
        with st.expander("📺 Additional Services"):
            col1, col2 = st.columns(2)
            with col1:
                multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
                online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
                device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            with col2:
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        
        # Predict button
        st.markdown("")
        predict_clicked = st.button("🔮 Predict Churn", use_container_width=True)
        
        if predict_clicked:
            # Prepare customer data
            customer_data = {
                "customerID": "WEB-001",
                "gender": gender,
                "SeniorCitizen": 1 if senior == "Yes" else 0,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": tenure * monthly_charges
            }
            
            # Make prediction (or simulate in demo mode)
            if demo_mode:
                # Simulate prediction based on risk factors
                risk_score = 0
                if contract == "Month-to-month":
                    risk_score += 0.3
                if tenure < 12:
                    risk_score += 0.2
                if payment == "Electronic check":
                    risk_score += 0.1
                if monthly_charges > 70:
                    risk_score += 0.1
                if internet == "Fiber optic" and online_security == "No":
                    risk_score += 0.1
                
                probability = min(risk_score + np.random.uniform(0, 0.2), 0.95)
                result = {
                    "probability": probability,
                    "churn_label": "Yes" if probability >= 0.5 else "No",
                    "risk_level": "High" if probability >= 0.7 else ("Medium" if probability >= 0.4 else "Low")
                }
            else:
                result = predictor.predict(customer_data)
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Gauge chart
                fig = create_gauge_chart(result["probability"])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Result summary
                st.markdown(f"""
                <div style="text-align: center; padding: 20px;">
                    <h4 style="color: #8b8b9a; margin-bottom: 10px;">CHURN PREDICTION</h4>
                    <h1 style="font-size: 48px; margin: 10px 0; color: {'#ff4757' if result['churn_label'] == 'Yes' else '#2ed573'};">
                        {result['churn_label'].upper()}
                    </h1>
                    {get_risk_badge(result['risk_level'])}
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                st.markdown("")
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Probability", f"{result['probability']:.1%}")
                with metric_col2:
                    st.metric("Confidence", f"{max(result['probability'], 1-result['probability']):.1%}")
            
            # Risk factors
            if not demo_mode:
                st.markdown("### ⚠️ Key Risk Factors")
                risk_factors = predictor.get_risk_factors(customer_data)
                
                for i, factor in enumerate(risk_factors, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="feature-card">
                            <strong>{i}. {factor['factor']}</strong>
                            <span style="float: right; color: {'#ff4757' if factor['impact'] == 'High' else '#ffa502'};">
                                {factor['impact']} Impact
                            </span>
                            <br><span style="color: #8b8b9a;">💡 {factor['recommendation']}</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                # Demo risk factors
                st.markdown("### ⚠️ Key Risk Factors")
                demo_factors = []
                if contract == "Month-to-month":
                    demo_factors.append(("Month-to-month contract", "High", "Offer contract upgrade incentives"))
                if tenure < 12:
                    demo_factors.append((f"New customer ({tenure} months)", "High", "Implement early engagement"))
                if payment == "Electronic check":
                    demo_factors.append(("Electronic check payment", "Medium", "Encourage auto-payment"))
                
                for i, (factor, impact, rec) in enumerate(demo_factors[:3], 1):
                    st.markdown(f"""
                    <div class="feature-card">
                        <strong>{i}. {factor}</strong>
                        <span style="float: right; color: {'#ff4757' if impact == 'High' else '#ffa502'};">
                            {impact} Impact
                        </span>
                        <br><span style="color: #8b8b9a;">💡 {rec}</span>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ==========================================================================
    # PAGE: BATCH ANALYSIS
    # ==========================================================================
    elif page == "📊 Batch Analysis":
        st.markdown("### Upload Customer Data")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with customer data",
            type=['csv'],
            help="File should contain customer attributes similar to the single prediction form"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            st.markdown(f"**Loaded {len(df)} customers**")
            
            with st.expander("📋 Preview Data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🚀 Run Batch Prediction", use_container_width=True):
                with st.spinner("Processing predictions..."):
                    if demo_mode:
                        # Simulate batch predictions
                        np.random.seed(42)
                        predictions = pd.DataFrame({
                            'prediction': np.random.randint(0, 2, len(df)),
                            'probability': np.random.uniform(0.1, 0.9, len(df))
                        })
                        predictions['risk_level'] = pd.cut(
                            predictions['probability'],
                            bins=[0, 0.4, 0.7, 1.0],
                            labels=['Low', 'Medium', 'High']
                        )
                        predictions['churn_label'] = predictions['prediction'].map({0: 'No', 1: 'Yes'})
                    else:
                        predictions = predictor.predict_batch(df)
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    churn_rate = predictions['prediction'].mean()
                    high_risk = (predictions['risk_level'] == 'High').sum()
                    
                    with col1:
                        st.metric("Total Customers", len(predictions))
                    with col2:
                        st.metric("Predicted Churn", f"{churn_rate:.1%}")
                    with col3:
                        st.metric("High Risk", high_risk)
                    with col4:
                        st.metric("Low Risk", (predictions['risk_level'] == 'Low').sum())
                    
                    # Distribution chart
                    st.markdown("### Risk Distribution")
                    
                    risk_counts = predictions['risk_level'].value_counts()
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=risk_counts.index,
                        values=risk_counts.values,
                        hole=0.6,
                        marker=dict(colors=['#2ed573', '#ffa502', '#ff4757'])
                    )])
                    
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': 'white'},
                        showlegend=True,
                        legend=dict(orientation="h", y=-0.1),
                        height=350
                    )
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Probability histogram
                        fig2 = px.histogram(
                            predictions, x='probability',
                            nbins=20,
                            color_discrete_sequence=['#667eea']
                        )
                        fig2.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': 'white'},
                            xaxis_title="Churn Probability",
                            yaxis_title="Count",
                            height=350
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Results table
                    st.markdown("### Detailed Results")
                    
                    # Combine with original data
                    results_df = pd.concat([df, predictions], axis=1)
                    
                    # Sort by probability
                    results_df = results_df.sort_values('probability', ascending=False)
                    
                    st.dataframe(
                        results_df[['customerID', 'probability', 'churn_label', 'risk_level']].head(20)
                        if 'customerID' in results_df.columns
                        else results_df[['probability', 'churn_label', 'risk_level']].head(20),
                        use_container_width=True
                    )
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "churn_predictions.csv",
                        "text/csv",
                        use_container_width=True
                    )
        else:
            # Sample data download
            st.info("💡 Don't have data? Download our sample template below.")
            
            sample_df = pd.DataFrame({
                'customerID': [f'CUST-{i:04d}' for i in range(5)],
                'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
                'SeniorCitizen': [0, 0, 1, 0, 0],
                'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
                'Dependents': ['No', 'No', 'Yes', 'No', 'Yes'],
                'tenure': [12, 3, 45, 8, 24],
                'PhoneService': ['Yes', 'Yes', 'Yes', 'No', 'Yes'],
                'InternetService': ['Fiber optic', 'DSL', 'No', 'Fiber optic', 'DSL'],
                'Contract': ['Month-to-month', 'Month-to-month', 'Two year', 'Month-to-month', 'One year'],
                'MonthlyCharges': [89.95, 45.50, 20.15, 75.00, 55.25],
                'TotalCharges': [1079.40, 136.50, 906.75, 600.00, 1326.00]
            })
            
            csv = sample_df.to_csv(index=False)
            st.download_button(
                "📥 Download Sample Template",
                csv,
                "sample_customers.csv",
                "text/csv"
            )
    
    # ==========================================================================
    # PAGE: MODEL INSIGHTS
    # ==========================================================================
    elif page == "📈 Model Insights":
        st.markdown("### Model Performance & Insights")
        
        if demo_mode:
            # Demo metrics
            metrics = {
                'accuracy': 0.86,
                'precision': 0.76,
                'recall': 0.68,
                'f1': 0.72,
                'roc_auc': 0.89
            }
            model_name = "XGBoost (Demo)"
        else:
            model_info = predictor.get_model_info()
            metrics = model_info.get('metrics', {})
            model_name = model_info.get('model_name', 'Unknown')
        
        # Model info
        st.markdown(f"**Current Model:** `{model_name}`")
        
        # Performance metrics
        st.markdown("### 📊 Performance Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.1%}")
        with col3:
            st.metric("Recall", f"{metrics.get('recall', 0):.1%}")
        with col4:
            st.metric("F1 Score", f"{metrics.get('f1', 0):.1%}")
        with col5:
            st.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.1%}")
        
        # Feature importance
        st.markdown("### 🎯 Feature Importance")
        
        if not demo_mode:
            try:
                from src.models.train import ModelTrainer
                trainer = ModelTrainer()
                trainer.load_model()
                importance_df = trainer.get_feature_importance()
            except:
                importance_df = None
        else:
            # Demo feature importance
            importance_df = pd.DataFrame({
                'feature': [
                    'Contract_Month-to-month', 'tenure', 'MonthlyCharges', 
                    'TotalCharges', 'InternetService_Fiber', 'PaymentMethod_Electronic',
                    'OnlineSecurity_No', 'TechSupport_No', 'SeniorCitizen', 'PaperlessBilling'
                ],
                'importance': [0.18, 0.17, 0.15, 0.14, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06]
            })
        
        if importance_df is not None:
            fig = create_feature_importance_chart(importance_df)
            st.plotly_chart(fig, use_container_width=True)
        
        # Methodology
        st.markdown("### 📚 Methodology")
        
        with st.expander("Learn more about the model"):
            st.markdown("""
            #### CRISP-DM Framework
            
            This project follows the Cross-Industry Standard Process for Data Mining:
            
            1. **Business Understanding**: Identify at-risk customers to enable proactive retention
            2. **Data Understanding**: Analyze customer demographics, services, and account info
            3. **Data Preparation**: Clean, encode, and normalize features
            4. **Modeling**: Train and compare multiple ML algorithms
            5. **Evaluation**: Select best model based on ROC-AUC score
            6. **Deployment**: Serve predictions via this Streamlit application
            
            #### Algorithms Compared
            
            - Logistic Regression
            - Random Forest
            - XGBoost
            - LightGBM
            
            #### Key Insights
            
            The most important factors for predicting churn are:
            - **Contract Type**: Month-to-month contracts have highest churn risk
            - **Tenure**: New customers (<12 months) are more likely to leave
            - **Monthly Charges**: Higher charges correlate with increased churn
            - **Payment Method**: Electronic check users have higher churn rates
            """)


if __name__ == "__main__":
    main()
