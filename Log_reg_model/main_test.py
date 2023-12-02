import os

import numpy as np
from matplotlib import pyplot as plt

from test_onePatient_logReg import testOnePatient
import pandas as pd

performance = np.zeros([40, 13])
patients = [30802, 80702]
seizures = [8, 6]
sop = [50, 40]
k = [5, 30]

for i in range(len(patients)):
    print("\n-//- Calculating for patient " + str(patients[i]) + " -//- ")
    performance[i, :] = testOnePatient(patients[i], sop[i], k[i], seizures[i])
    np.save("TestingResultsLR.npy", performance)


