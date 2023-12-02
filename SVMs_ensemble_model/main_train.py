import os

import numpy as np
from train_onePatient_SVMs import calculatePreIctalAndFeatureNumber
import pandas as pd

training = np.zeros([6, 4])
patients = [30802, 80702, 98202]


for i in range(len(patients)):
    print("\n-//- Calculating for patient " + str(patients[i]) + " -//- ")
    training[i, :] = calculatePreIctalAndFeatureNumber(patients[i])
    np.save("TrainingResultsSVM.npy", training)
