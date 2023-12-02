from getPlots_SNNs import getPlots
import matplotlib.pyplot as plt

patients = []
sop = []
seizures = []

for i in range(0, len(patients)):
    print("\n-//- Plotting for patient " + str(patients[i]) + " -//- ")
    getPlots(patients[i], sop[i], seizures[i])
    plt.close('all')
    