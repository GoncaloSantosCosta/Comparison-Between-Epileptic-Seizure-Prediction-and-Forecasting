import time

import tensorflow as tf
from train_onePatient_SNN import train_model_SNN
import numpy as np

sops = np.zeros([40, 2])
patients = [30802]

for i in range(0, len(patients)):
    print("\n-//- Calculating for patient " + str(patients[i]) + " -//- ")
    sops[i, :] = train_model_SNN(patients[i])
    np.save('TrainingResultsSNN24', sops)
