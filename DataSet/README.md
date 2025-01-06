# Dataset Description

This folder contains the training and test datasets for the InterDIA project.

## Dataset Overview
- Total compounds: 597 drugs
  - Training set: 477 drugs (118 DIA-positive, 359 DIA-negative)
  - Test set: 120 drugs (30 DIA-positive, 90 DIA-negative)

## File Structure
```
DataSet/
├── DIA_trainingset_RDKit_descriptors.csv   # Training set with RDKit descriptors 
├── DIA_testset_RDKit_descriptors.csv       # Test set with RDKit descriptors
├── DIA_trainingset_DS_descriptors.csv      # Training set with Discovery Studio descriptors
├── DIA_testset_DS_descriptors.csv          # Test set with Discovery Studio descriptors 
├── DIA_trainingset_MOE_descriptors.csv     # Training set with MOE descriptors
├── DIA_testset_MOE_descriptors.csv         # Test set with MOE descriptors
├── DIA_trainingset_Mold2_descriptors.csv   # Training set with Mold2 descriptors
├── DIA_testset_Mold2_descriptors.csv       # Test set with Mold2 descriptors
├── DIA-TrainingSet.sdf                     # Training set molecular structures
└── DIA-TestSet.sdf                         # Test set molecular structures
```

## Data Format
Each CSV file contains the following columns:
1. Label: Binary classification (1 for DIA-positive, 0 for DIA-negative)
2. SMILES: Canonical SMILES representation of molecular structure
3. RDKit descriptors: 196 molecular descriptors calculated using RDKit (http://www.scbdd.com/rdk_desc/index/)
