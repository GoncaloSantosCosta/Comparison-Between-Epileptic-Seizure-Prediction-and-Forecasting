import os

import numpy as np
from matplotlib import pyplot as plt

from test_onePatient_SVMs import testOnePatient
import pandas as pd

performance = np.zeros([40, 13])
patients = [80702]
seizures = [6]
sop = [45]
k = [15]
c_value = [0.0009765625]

for i in range(len(patients)):
    print("\n-//- Calculating for patient " + str(patients[i]) + " -//- ")
    performance[i, :] = testOnePatient(patients[i], sop[i], k[i], seizures[i], c_value[i])
    np.save("TestingResultsSVM.npy", performance)
