
# InterDIA: Interpretable Prediction of Drug-induced Autoimmunity through Ensemble Machine Learning
## Streamlit Web Application (https://drug-induced-autoimmunity-predictor.streamlit.app/)

### File Structure
```
webapp/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Streamlit-specific dependencies
├── scaler_and_model.pkl    # Saved model and scaler
├── Xtrain_std.pkl          # Standardized training data

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

### Deployment Guide
1. **Connect GitHub Account**

Visit Streamlit Cloud
Click "Sign in with GitHub"
Authorize Streamlit to access your repositories

2. **Create New Application**

Click "New app" after logging in
You'll see the "Deploy an app" page

3. **Configure Deployment**

Select your GitHub repository
Choose your branch (main or master)
Enter "webapp/app.py" as the Main file path

4. **Launch Application**

Click "Deploy"
Streamlit will automatically:

Install required dependencies
Launch the application
Generate a public URL

5. **Access Application**

Use the provided URL to access your app
Format: https://your-app-name.streamlit.app

6. **Maintenance**

Updates to GitHub automatically trigger redeployment
No manual deployment needed
Monitor the deployment logs for any issues

**Important Notes**

Ensure all paths in app.py are relative;
Include all dependencies in requirements.txt;
Verify model files are properly uploaded;
Test locally before deployment.
