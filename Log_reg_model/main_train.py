import os

import numpy as np
import time
from train_onePatient_logReg import calculatePreIctalAndFeatureNumber
import pandas as pd

training = np.zeros([40, 3])
patients = [30802, 80702, 98202]


for i in range(len(patients)):
    print("\n-//- Performing grid search for patient " + str(patients[i]) + " -//- ")
    training[i, :] = calculatePreIctalAndFeatureNumber(patients[i])
    np.save("TrainingResultsLR.npy", training)


