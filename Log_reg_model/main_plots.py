from getPlots_logReg import getPlots
import matplotlib.pyplot as plt

patients = [30802, 80702]
seizures = [8, 6]
sop = [50, 40]
k = [5, 30]

for i in range(0, len(patients)):
    print("\n-//- Plotting for patient " + str(patients[i]) + " -//- ")
    getPlots(patients[i], sop[i], k[i], seizures[i])
    plt.close('all')
