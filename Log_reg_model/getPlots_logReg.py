import os

import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime
import numpy as np
import utils

from test_OnePatient_getPlots_logReg import testOnePatientGetPlots


def getPlots(patient, sop, k, seizures):
    # get classifier time plots
    [datetimes, labels, alarms, firing_power_output,
     vigilance, vigilance_datetimes] = testOnePatientGetPlots(patient, sop, k, seizures)
    
    os.chdir("Results")

    new_datetimes = []
    for i in range(seizures - 3):
        new_datetimes_i = []
        for j in range(len(datetimes[i])):
            new_datetimes_i.append(utils.convertIntoDatetime(datetimes[i][j]))
        new_datetimes.append(new_datetimes_i)
    datetimes = new_datetimes

    datetimes_new = []
    firing_power_output_new = []
    labels_new = []
    alarms_new = []

    # filling missing data
    for i in range(len(datetimes)):
        labels[i] = labels[i].tolist()
        firing_power_output[i] = firing_power_output[i].tolist()

        datetimes_new_i = []
        firing_power_output_new_i = []
        labels_new_i = []
        alarms_new_i = []

        for j in range(len(datetimes[i]) - 1):
            time_difference = datetimes[i][j + 1] - datetimes[i][j]
            time_difference = time_difference.seconds

            datetimes_new_i.append(datetimes[i][j])
            firing_power_output_new_i.append(firing_power_output[i][j])
            labels_new_i.append(labels[i][j])
            alarms_new_i.append(alarms[i][j])

            if time_difference <= 5:
                pass
            else:
                new_datetime = datetimes[i][j] + datetime.timedelta(0, 5)
                while time_difference > 5:
                    datetimes_new_i.append(new_datetime)
                    labels_new_i.append(np.NaN)
                    alarms_new_i.append(np.NaN)
                    firing_power_output_new_i.append(np.NaN)

                    time_difference = datetimes[i][j + 1] - new_datetime
                    time_difference = time_difference.seconds
                    new_datetime = new_datetime + datetime.timedelta(0, 5)

        datetimes_new.append(datetimes_new_i)
        firing_power_output_new.append(firing_power_output_new_i)
        labels_new.append(labels_new_i)
        alarms_new.append(alarms_new_i)

    datetimes = datetimes_new
    firing_power_output = firing_power_output_new
    labels = labels_new
    alarms = alarms_new

    # plotting postprocessing output throughout time
    for i in range(seizures - 3):

        fig, (pred, fore) = plt.subplots(2, 1, figsize=(16, 14))
        fig.suptitle("Patient " + str(patient) + ", Seizure " + str(i + 3 + 1))

        ########## PREDICTION ##########
        pred.plot(datetimes[i], firing_power_output[i], 'k', alpha=0.7)
        pred.plot(datetimes[i], np.linspace(0.7, 0.7, len(datetimes[i])), linestyle='--',
                  color='black', alpha=0.7)

        pred.grid()

        pred.set_ylim(0, 1)
        pred.set_xlim(datetimes[i][0], datetimes[i][len(datetimes[i]) - 1])

        for alarm_index in np.where(np.diff(alarms[i]) == 1)[0]:
            pred.plot(datetimes[i][alarm_index], firing_power_output[i][alarm_index],
                      marker='^', color='maroon', markersize=10)

        if len(np.where(np.diff(alarms[i]) == 1)[0]) > 0:
            pred.plot([], [], marker='^', color='maroon', markersize=10, label="Alarm")

        if 1 in labels[i]:
            pred.fill_between(datetimes[i], 0, 1, where=np.array(datetimes[i]) >
                                                        np.array(datetimes[i][np.where(np.diff(labels[i]) == 1)[0][0]]),
                              facecolor='moccasin', alpha=0.5, label="Preictal Period")

            pred.axvline(x=datetimes[i][np.where(np.diff(labels[i]) == 1)[0][0]], color='k',
                         alpha=0.7, linestyle='--', linewidth=0.8)

        pred.legend()
        pred.set_title("Prediction")

        xfmt = md.DateFormatter('%H:%M:%S')
        pred.xaxis.set_major_formatter(xfmt)
        pred.yaxis.set_ticks([0, 0.05, 0.2, 0.4, 0.6, 0.7, 0.8, 0.95, 1.0])
        pred.yaxis.set_ticklabels(["0", "sleep", "0.2", "0.4", "0.6", "0.7", "0.8", "awake", "1.0"])
        
        if patient in [8902, 60002, 93902, 123902]:
            pred.plot(vigilance_datetimes[i], vigilance[i], alpha=0.4)
        else:
            pred.plot(vigilance_datetimes[i + 3], vigilance[i + 3], alpha=0.4)

        ########## FORECASTING ##########
        fore.plot(datetimes[i], firing_power_output[i], 'k', alpha=0.7)
        fore.plot(datetimes[i], np.linspace(0.7, 0.7, len(datetimes[i])), linestyle='--',
                  color='black', alpha=0.7)
        fore.plot(datetimes[i], np.linspace(0.3, 0.3, len(datetimes[i])), linestyle='--',
                  color='black', alpha=0.7)

        fore.grid()

        fore.set_ylim(0, 1)
        fore.set_xlim(datetimes[i][0], datetimes[i][len(datetimes[i]) - 1])

        fore.fill_between(datetimes[i], np.array(firing_power_output[i]), where=np.array(firing_power_output[i]) > 0.7,
                          facecolor='brown', alpha=0.5, label="High")

        fore.fill_between(datetimes[i], np.array(firing_power_output[i]),
                          where=(0.3 <= np.array(firing_power_output[i])) &
                                (np.array(firing_power_output[i]) <= 0.7),
                          facecolor='yellow', alpha=0.4, label="Moderate")

        fore.fill_between(datetimes[i], np.array(firing_power_output[i]), where=np.array(firing_power_output[i]) < 0.3,
                          facecolor='lightgreen', alpha=0.5, label="Low")

        if 1 in labels[i]:
            fore.fill_between(datetimes[i], 0, 1, where=np.array(datetimes[i]) >
                                                        np.array(datetimes[i][np.where(np.diff(labels[i]) == 1)[0][0]]),
                              facecolor='moccasin', alpha=0.5, label="Preictal Period")

            fore.axvline(x=datetimes[i][np.where(np.diff(labels[i]) == 1)[0][0]], color='k',
                         alpha=0.7, linestyle='--', linewidth=0.8)

        fore.legend()
        fore.set_title("Forecasting")

        fore.xaxis.set_major_formatter(xfmt)
        fore.yaxis.set_ticks([0, 0.05, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.95, 1.0])
        fore.yaxis.set_ticklabels(["0", "sleep", "0.2", "0.3", "0.4", "0.6", "0.7", "0.8", "awake", "1.0"])

        if patient in [8902, 60002, 93902, 123902]:
            fore.plot(vigilance_datetimes[i], vigilance[i], alpha=0.4)
        else:
            fore.plot(vigilance_datetimes[i + 3], vigilance[i + 3], alpha=0.4)

        # plt.show()
        fig.savefig("Patient" + str(patient) + "Seizure" + str(i + 3 + 1) + "LR.png", dpi=200, bbox_inches='tight')
        fig.savefig("Patient" + str(patient) + "Seizure" + str(i + 3 + 1) + "LR.pdf", dpi=200, bbox_inches='tight')

    os.chdir("..")
