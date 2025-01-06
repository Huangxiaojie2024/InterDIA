import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Page Title and Introduction
st.set_page_config(
    page_title="Drug-induced Autoimmunity (DIA) Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styles
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            width: 100%;
            background-color: #0d6efd;
            color: white;
            border-radius: 5px;
            padding: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .stProgress .st-bo {
            background-color: #0d6efd;
        }
        .plot-container {
            background-color: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #0d6efd;
            text-align: center;
            padding: 1rem;
        }
        h2 {
            color: #0d6efd;
            padding: 0.5rem 0;
        }
        .stAlert {
            background-color: #e7f3fe;
            border-left-color: #0d6efd;
        }
        div[data-testid="stMetricValue"] {
            font-size: 24px;
            color: #0d6efd;
        }
        .metric-container {
            background-color: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        .shap-container {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Load Model and Scaler
@st.cache_resource
def load_model():
    with open('scaler_and_model.pkl', 'rb') as f:
        scaler, best_estimator_eec = pickle.load(f)
    with open('Xtrain_std.pkl', 'rb') as f:
        Xtrain_std = pickle.load(f)
    return scaler, best_estimator_eec, Xtrain_std

scaler, best_estimator_eec, Xtrain_std = load_model()

# List of 65 Optimal Molecular Descriptor Names
descriptor_names = ['BalabanJ', 'Chi0', 'EState_VSA1', 'EState_VSA10', 'EState_VSA4', 'EState_VSA6', 
                    'EState_VSA9', 'HallKierAlpha', 'Ipc', 'Kappa3', 'NHOHCount', 'NumAliphaticHeterocycles',
                    'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticRings', 'PEOE_VSA10',
                    'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA6', 'PEOE_VSA7', 
                    'PEOE_VSA9', 'RingCount', 'SMR_VSA10', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA9', 
                    'SlogP_VSA10', 'SlogP_VSA5', 'SlogP_VSA8', 'VSA_EState8', 'fr_ArN', 'fr_Ar_NH', 'fr_C_O', 
                    'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_alkyl_carbamate', 'fr_allylic_oxid', 'fr_amide', 
                    'fr_aryl_methyl', 'fr_azo', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_dihydropyridine', 'fr_epoxide', 
                    'fr_ether', 'fr_furan', 'fr_guanido', 'fr_hdrzone', 'fr_imide', 'fr_ketone_Topliss', 'fr_lactam', 
                    'fr_methoxy', 'fr_morpholine', 'fr_nitro_arom', 'fr_para_hydroxylation', 'fr_phos_ester', 'fr_piperdine', 
                    'fr_pyridine', 'fr_sulfide', 'fr_term_acetylene', 'fr_unbrch_alkane']

# Page Title and Introduction
st.title("🔬 Drug-induced Autoimmunity (DIA) Predictor")
st.markdown("""
    <div style='background-color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <p style='font-size: 1.1em; color: #666;'>
            Welcome to the DIA Predictor, an advanced machine learning-based tool for predicting drug-induced autoimmunity. 
            This tool utilizes molecular descriptors to assess the potential risk of drugs causing autoimmune disease.
        </p>
    </div>
""", unsafe_allow_html=True)
# Sidebar Design
with st.sidebar:
    st.header("📊 Data Input")
    uploaded_file = st.file_uploader("Upload RDKit descriptors CSV", type=['csv'])
    
    if uploaded_file:
        st.success(f"File uploaded successfully!")
        
    st.markdown("---")
    st.markdown("""
        ### 📖 Instructions
        1. Visit http://www.scbdd.com/rdk_desc/index/ to calculate and download 196 RDKit molecular descriptors for your compounds
        2. The model will automatically select and use the 65 optimal descriptors for prediction
        3. Upload the descriptors file using the button above
        4. View predictions and analysis in the main panel
    """)
    
    st.markdown("---")
    st.markdown("""
        ### 🎯 Model Information
        - **Algorithm**: Easy Ensemble Classifier
        - **Input Features**: 65 selected RDKit molecular descriptors
        - **Output**: Binary classification (DIA positive/negative)
        - **Data Source**: RDKit descriptors calculated from http://www.scbdd.com/rdk_desc/index/
    """)

# Main Content
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Data Validation
        missing_descriptors = [desc for desc in descriptor_names if desc not in df.columns]
        if missing_descriptors:
            st.error(f"Missing required descriptors: {', '.join(missing_descriptors)}")
        else:
            # Data Processing and Predictions
            X = df[descriptor_names].values
            X_std = scaler.transform(X)
            predictions_prob = best_estimator_eec.predict_proba(X_std)
            
            # Create Results DataFrame
            results_df = pd.DataFrame({
                "Compound_ID": range(1, len(df) + 1),
                "DIA_negative_prob": predictions_prob[:, 0],
                "DIA_positive_prob": predictions_prob[:, 1],
                "Prediction": ["DIA Positive" if p > 0.5 else "DIA Negative" for p in predictions_prob[:, 1]],
                "Risk_Level": pd.cut(predictions_prob[:, 1], 
                                   bins=[0, 0.2, 0.5, 0.8, 1.0],
                                   labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
            })
            
            # Display Key Metrics
            st.subheader("📊 Prediction Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                positive_count = sum(predictions_prob[:, 1] > 0.5)
                st.metric("DIA Positive", 
                         f"{positive_count}",
                         f"{positive_count/len(df)*100:.1f}%")
            
            with col2:
                high_risk = sum(predictions_prob[:, 1] > 0.8)
                st.metric("High Risk",
                         f"{high_risk}",
                         f"{high_risk/len(df)*100:.1f}%")
            
            with col3:
                med_risk = sum((predictions_prob[:, 1] > 0.5) & (predictions_prob[:, 1] <= 0.8))
                st.metric("Medium Risk",
                         f"{med_risk}",
                         f"{med_risk/len(df)*100:.1f}%")
            
            with col4:
                avg_prob = np.mean(predictions_prob[:, 1])
                st.metric("Avg. Probability",
                         f"{avg_prob:.3f}",
                         f"±{np.std(predictions_prob[:, 1]):.3f}")

            # Risk Distribution Visualization
            st.subheader("📈 Risk Distribution Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie Chart for Prediction Distribution
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['DIA Positive', 'DIA Negative'],
                    values=[positive_count, len(df) - positive_count],
                    hole=.3,
                    marker_colors=['#ff6b6b', '#4ecdc4']
                )])
                fig_pie.update_layout(
                    title="Prediction Distribution",
                    showlegend=True,
                    height=400
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Bar Chart for Risk Levels
                risk_counts = results_df['Risk_Level'].value_counts().sort_index()
                fig_risk = go.Figure(data=[go.Bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    marker_color=['#4ecdc4', '#ffe66d', '#ff9f1c', '#ff6b6b']
                )])
                fig_risk.update_layout(
                    title="Risk Level Distribution",
                    xaxis_title="Risk Level",
                    yaxis_title="Number of Compounds",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_risk, use_container_width=True)

            # Show Detailed Results Table
            st.subheader("📋 Detailed Results")
            st.dataframe(results_df.style.background_gradient(
                subset=['DIA_positive_prob'],
                cmap='RdYlBu_r'
            ))
            
            # Download Results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Complete Results",
                data=csv,
                file_name="dia_predictions.csv",
                mime="text/csv"
            )
            
            # SHAP Analysis
            st.markdown("---")
            st.subheader("🔍 SHAP Analysis")
            
            # Select Compound
            selected_compound = st.selectbox(
                "Select a compound for detailed analysis:",
                range(len(df)),
                format_func=lambda x: (
                    f"Compound {x+1} "
                    f"({'🔴' if predictions_prob[x,1] > 0.5 else '🟢'}) "
                    f"Risk: {results_df.iloc[x]['Risk_Level']} "
                    f"(P={predictions_prob[x,1]:.3f})"
                )
            )

            if selected_compound is not None:
                st.session_state.selected_compound = selected_compound
                
                # Display Selected Compound Prediction Details
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Prediction", 
                        "DIA Positive" if predictions_prob[selected_compound,1] > 0.5 else "DIA Negative",
                        f"P={predictions_prob[selected_compound,1]:.3f}"
                    )
                with col2:
                    st.metric(
                        "Prediction Probability",
                         f"{predictions_prob[selected_compound,1]*100:.2f}%"
                         )
                with col3:
                    st.metric(
                        "Risk Level",
                        results_df.iloc[selected_compound]['Risk_Level']
                    )
               
                
                # SHAP Analysis
                with st.spinner('Analyzing molecular features...'):
                    explainer = shap.KernelExplainer(
                        best_estimator_eec.predict_proba,
                        Xtrain_std
                    )
                    
                    # Set Random Seed
                    np.random.seed(1)
                    
                    # Calculate SHAP Values
                    shap_values = explainer.shap_values(
                        X_std[selected_compound:selected_compound+1],
                        nsamples=150  # Increase samples for stability
                    )
                    
                    # SHAP Waterfall Plot
                    st.markdown("### SHAP Waterfall Plot")
                    col1, col2, col3 = st.columns([1,6,1])
                    
                    with col2:
                        # Inverse Standardize Current Sample
                        sample_original = scaler.inverse_transform(X_std[selected_compound:selected_compound+1])
                        
                        # Create Waterfall Plot
                        fig, ax = plt.subplots(figsize=(8, 6))
                        shap.waterfall_plot(
                            shap.Explanation(
                                values=shap_values[0,:,1],  # Use SHAP values for positive class
                                base_values=explainer.expected_value[1],  # Model baseline value
                                data=sample_original[0],  # Inverse standardized feature values
                                feature_names=descriptor_names  # Feature names
                            ),
                            show=False,
                            max_display=10  # Display top 10 features
                        )
                        plt.title("Impact of Features on Model Prediction")
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                    
                    st.success('Analysis completed successfully!')

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    # Display Welcome Message
    st.info("👆 Please upload your RDKit descriptors CSV file to begin the analysis.")
    
    # Add Example Instructions
    st.markdown("""
        ### 📝 Data Requirements
        1. Calculate RDKit descriptors from http://www.scbdd.com/rdk_desc/index/
        2. The CSV file containing RDKit molecular descriptors can be directly uploaded
        3. The model will automatically select the 65 optimal descriptors
        
        ### 🎯 Key Features
        - Advanced Easy Ensemble Classifier for DIA prediction
        - Interactive SHAP explanations
        - Risk level assessment
    """)
