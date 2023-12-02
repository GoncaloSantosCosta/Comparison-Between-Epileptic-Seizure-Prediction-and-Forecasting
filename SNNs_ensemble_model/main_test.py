import os
import numpy as np
from matplotlib import pyplot as plt

from test_onePatient_SNN import testOnePatient


performance = np.zeros([40, 13])
patients = [30802, 80702, 98202]
seizures = [8, 6, 8]
sop = [40, 35, 35]


for i in range(0, len(patients)):
    print("\n-//- Calculating for patient " + str(patients[i]) + " -//- ")
    performance[i, :] = testOnePatient(patients[i], sop[i], seizures[i])
    # np.save("TestingResultsSNN.npy", performance)

# plt.savefig("RL_LR.pdf", dpi=200, bbox_inches='tight')
# plt.savefig("RL_LR.png", dpi=200, bbox_inches='tight')
