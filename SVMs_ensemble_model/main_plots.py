from getPlots_SVMs import getPlots
import matplotlib.pyplot as plt


patients = []
seizures = []
sop = []
k = []
c_value = []

for i in range(0, len(patients)):
    print("\n-//- Plotting for patient " + str(patients[i]) + " -//- ")
    getPlots(patients[i], sop[i], k[i], c_value[i], seizures[i])
    plt.close('all')
