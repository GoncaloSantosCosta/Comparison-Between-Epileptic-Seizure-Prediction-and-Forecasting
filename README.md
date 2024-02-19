## Comparison-Between-Epileptic-Seizure-Prediction-and-Forecasting

This is the python code used in "Comparison Between Epileptic Seizure Prediction and Forecasting Based on Machine Learning", currently i review in Scientific Reports. This is the full code for three seizure prediction and forecasting Machine Learning frameworks (the classifiers for each framework are a Logistic Regression, an Ensemble of 15 Support Vector Machines (SVMs), and an ensemble of 15 Shallow neural Networks (SNNs)). It also provides scripts to plot the postprocessing output in time. The difference between prediction and forecasting in this study was the postprocessing method used, so each pipeline shows results for both approaches.

This study uses data from the EPILEPSIAE database. The data was provided under license for this study by the EPILEPSIAE Consortium. Therefore, it is not publicly available. However, it can be made available upon reasonable request and with permission from the EPILEPSIAE Consortium.

Three folders are available, one for each of the pipelines we entioned above.

#### Main files
These files are the ones you execute:

- [main_train.py]: execute it to perform a grid-search to find the optimal hyperparameters (preictal period, k number of features, and/or SVM C value).
- [main_test.py]: execute it to train and test the model using the optimal hyperparameters.
- [main_plots.py]: execute it to plot the Firing Power over time.

#### Function files
These files contain function that are mostly specific to each pipeline:

- [train_onePatient_logReg.py]: function to perform a grid-search and find the optimal hyperparameters for the Logistic Regression (preictal period and k number of features).
- [train_onePatient_SVMs.py]: function to perform a grid-search and find the optimal hyperparameters for the 15 SVM ensemble (preictal period, k number of features, and SVM C value).
- [train_onePatient_SNNs.py]: function to perform a grid-search and find the optimal hyperparameters for the 15 SNN ensemble (preictal period).
- [train_SNN.py]: function to construct the SNN architecture, train it and save it.
  
- [test_onePatient_logReg.py]: function train and test the Logistic Regression.
- [test_onePatient_SVMs.py]: function train and test the 15 SVM ensemble.
- [test_onePatient_SNNs.py]: function train and test the 15 SNN ensemble.
  
- [test_onePatient_getPlots_logReg.py]: function train and test the Logistic Regression that returns the neccessary information to plot the Firing Power over time.
- [test_onePatient_getPlots_SVMs.py]: function train and test the 15 SVM ensemble that returns the neccessary information to plot the Firing Power over time.
- [test_onePatient_getPlots_SNNs.py]: function train and test the 15 SNN ensemble that returns the neccessary information to plot the Firing Power over time.
- [getPlots_logReg.py]: function Firing Power over time using the Logistic Regression.
- [getPlots_SVMs.py]: function Firing Power over time using the 15 SVM ensemble.
- [getPlots_SNNs.py]: function Firing Power over time using the 15 SNN ensemble.

- [utils.py]: script containing several utility functions used throughout the pipeline.
