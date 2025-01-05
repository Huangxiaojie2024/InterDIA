
# InterDIA: Interpretable Prediction of Drug-induced Autoimmunity through Ensemble Machine Learning
## Streamlit Web Application

### File Structure
```
webapp/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Streamlit-specific dependencies
├── models/
│   ├── scaler_and_model.pkl    # Saved model and scaler
│   └── Xtrain_std.pkl          # Standardized training data

```

### Required Files
The application requires the following model files:
- `scaler_and_model.pkl`: Contains the trained EEC model and feature scaler
- `Xtrain_std.pkl`: Standardized training data
- `requirements.txt`: Specific dependencies for the web application

### Running the Application
```bash
cd webapp
streamlit run app.py
```

### Features
1. **Interactive Interface**
   - Upload RDKit descriptor CSV files
   - Real-time prediction visualization
   - Risk level assessment

2. **Prediction Analysis**
   - Binary classification (DIA positive/negative)
   - Probability scores
   - Risk level distribution
   - Detailed results table

3. **SHAP Analysis**
   - Interactive feature importance visualization
   - Individual prediction explanation
   - Waterfall plots for molecular features

### Usage Instructions
1. Visit http://www.scbdd.com/rdk_desc/index/ to calculate RDKit descriptors
2. Upload the CSV file containing the descriptors
3. View predictions and analysis results
4. Download detailed prediction results

### Model Details
The web application uses:
- 65 optimized RDKit molecular descriptors
- Easy Ensemble Classifier for predictions
- SHAP (SHapley Additive exPlanations) for interpretation

### Input Format
The input CSV should contain the RDKit descriptors

