# Parkinson's Disease Classification Using XGBoost

## Introduction
This project focuses on classifying individuals as healthy or having Parkinson's Disease (PD) based on vocal feature data. The aim is to develop a predictive model using the XGBoost algorithm to facilitate early diagnosis and monitoring of PD.

## Data Description
The project utilizes two datasets:
1. **parkinsons.data**: Contains features derived from vocal recordings, such as fundamental frequency and various voice quality measures.
2. **parkinsons_updrs.data**: Provides additional clinical measurements including motor and total UPDRS scores.

### Key Attributes
- **Fundamental Frequency Measurements**: Fo, Fhi, Flo
- **Voice Quality Measures**: Jitter, Shimmer, NHR, HNR
- **Status**: Indicates if the subject is diagnosed with Parkinson's Disease (1) or is healthy (0).

## Methodology
### Data Preparation
1. **Loading Data**: Both datasets are read into pandas DataFrames.
2. **Data Merging**: Common columns are identified and merged into a single DataFrame.
3. **Random Data Generation**: Random samples are generated based on the statistical properties of the dataset for testing purposes.
4. **Feature Selection**: Features are selected by dropping the 'status' and 'name' columns.

### Data Scaling
Features are scaled to a range between -1 and 1 using Min-Max scaling.

### Model Training
1. **Train-Test Split**: The dataset is split into training and testing sets (80-20).
2. **Model Initialization**: An XGBoost classifier is created with a learning rate of 0.1.
3. **Model Training**: The model is trained using the training dataset.

### Model Evaluation
The model's accuracy is calculated using the testing dataset, and results are printed.

### Model and Scaler Saving
The trained model and the scaler used for feature scaling are saved as pickle files for future use.

### Results
The model achieved an accuracy of 93% on the test dataset, indicating its effectiveness in distinguishing between healthy individuals and those with Parkinson's Disease.

### Conclusion
This project successfully demonstrates the application of XGBoost for classifying Parkinson's Disease using vocal feature data. Future work could focus on hyperparameter tuning, feature selection, and cross-validation to enhance model performance.