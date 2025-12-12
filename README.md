# 00_READ_ME

## Table of Contents
- [01_Analysis.py](#01_analysispy)
  - 0\. Setup and Helper Functions
  - 1\. Data Loading and Within-Dataset Predictions
  - 2\. Permutation Importance Analysis
  - 3\. Across-Dataset Predictions
  - 4\. Statistical Model Comparisons
  - 5\. Subgroup Analysis
  - 6\. Post-Hoc Matched Samples Analysis
  - Computational Note
- [02_Figures.py](#02_figurespy)
- Data Access Notes
  - ADNI
  - OASIS-3

---

## 01_Analysis.py

**Summary:**  
`01_Analysis.py` contains code for predictive modeling and statistical analyses comparing cognitive-decline predictions across the OASIS-3 and ADNI datasets.

---

### 0. Setup and Helper Functions
- 0.1 Package Imports  
- 0.2 Helper Functions for Data Processing, Model Fitting, and Evaluation

---

### 1. Data Loading and Within-Dataset Predictions
- 1.1 Data Loading and Preprocessing  
- 1.2 Within-Dataset Predictions (OASIS-3 and ADNI)  
- 1.3 Results Aggregation and Summary  

---

### 2. Permutation Importance Analysis
- 2.1 Permutation Importance for Individual Features  
- 2.2 Feature-Importance Visualization and Analysis  
- 2.3 Coalition-Based Feature Importance  

---

### 3. Across-Dataset Predictions
- 3.1 Cross-Dataset Prediction (Clinical, Structural, Combined, Top-15)  
- 3.2 Cross-Dataset Results Aggregation  

---

### 4. Statistical Model Comparisons
- 4.1 Statistical Testing Framework and Model Comparisons  
- 4.2 Absolute Error Comparisons Between Models  

---

### 5. Subgroup Analysis
- 5.1 Absolute Error Analysis Preparation  
- 5.2 Subgroup Comparisons by Diagnosis, Sex, APOE ε4, and Age  
- 5.3 Results Formatting and Export  

---

### 6. Post-Hoc Matched Samples Analysis
- 6.1 Quantile-Based Sample Matching  
- 6.2 Matched Sample Predictions and Evaluation  
- 6.3 Results for Combined and Top-15 Models  

---

### ⚠️ Computational Note
Running the predictive models is computationally expensive (`n_splits = 1000`).  
Consider reducing the number of splits for testing.

---

## 02_Figures.py

`02_Figures.py` generates all figures used in the research article:

- Figure 1  
- Figure 2  
- Supplementary Figures B–F  
- *(Supplementary Figure G is produced via `01_Analysis.py`, Section 2.3.)*

---

## Data Access Notes

### ADNI
Part of the data used in preparation for this article was obtained from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) database (https://adni.loni.usc.edu). Access to the publicly available ADNI data requires submitting an application through the Laboratory of Neuro Imaging (LONI) Image and Data Archive (IDA):  
https://adni.loni.usc.edu/data-samples/access-data/

### OASIS-3
OASIS-3 is an open-access longitudinal multimodal neuroimaging, clinical, and cognitive dataset for normal aging and Alzheimer’s disease. Access the dataset here:  
https://www.oasis-brains.org
