# InterDIA: Interpretable Prediction of Drug-induced Autoimmunity through Ensemble Machine Learning

## Overview
InterDIA is a comprehensive and interpretable machine learning framework for predicting drug-induced autoimmunity (DIA) toxicity. By integrating state-of-the-art ensemble learning approaches with multi-strategy feature selection, this framework provides accurate predictions while offering mechanistic insights through SHAP (SHapley Additive exPlanations) analysis.
![Research Workflow](figures/workflow.png)
## Key Features
- Advanced ensemble resampling techniques for handling imbalanced data
- Multi-strategy feature selection approaches
- Interpretable predictions through SHAP analysis
- Support for batch processing
- Free online prediction platform

## Dataset
The dataset comprises 597 drugs (148 DIA-positive and 449 DIA-negative):
- Training set: 477 drugs (118 positive, 359 negative)
- External validation set: 120 drugs (30 positive, 90 negative)

## Installation

### Requirements
Python 3.9.19 is required for running this project. We recommend using a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
Please check requirements.txt for detailed dependency information. Main packages include:

- Core Dependencies
  - numpy==1.23.5
  - pandas==2.2.3
  - scipy==1.11.2

- Machine Learning Libraries
  - scikit-learn==1.5.1
  - xgboost==1.6.1
  - lightgbm==3.3.5
  - imbalanced-learn==0.12.3
  - deap==1.4.1
  - hyperopt==0.2.7

- Visualization
  - matplotlib==3.9.2
  - seaborn==0.13.2
  - shap==0.46.0
  - plotly==5.24.1

## Code Structure
```
.
├── data/
│   ├── DIA_trainingset_RDKit_descriptors.csv     # Training set with RDKit descriptors
│   └── DIA_testset_RDKit_descriptors.csv         # Test set with RDKit descriptors
├── notebooks/
│   └── DIA_RDKit_Prediction.ipynb                # Main analysis notebook using RDKit descriptors

├── streamlit/
│   ├── app.py                                # Streamlit web application
│   ├── requirements.txt                      # Streamlit-specific requirements
│   └── utils/                               # Utility functions for web app
├── figures/
│   └── workflow.png                              # Research workflow diagram
├── requirements.txt
└── README.md
```

## Citations
If you use this code in your research, please cite:

```
Huang, L., Liu, P., & Huang, X. (2024). InterDIA: Interpretable Prediction of Drug-induced 
Autoimmunity through Ensemble Machine Learning. 
```

## Contributing
We welcome contributions to this project. Please feel free to submit issues and pull requests.

## Contact
For questions and feedback, please contact:
- Xiaojie Huang - huangxj46@mail3.sysu.edu.cn

## Acknowledgments
This research was supported by the Medical Science and Technology Research Foundation of Guangdong Province (Grant Number: A2024082).
