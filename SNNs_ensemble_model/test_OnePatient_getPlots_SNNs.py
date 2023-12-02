import os
import numpy as np
import utils

from keras.models import Model
from keras.layers import *
import tensorflow as tf

# due to the np.delete function
import warnings

warnings.filterwarnings("ignore")


def testOnePatientGetPlots(patient, sop, total_seizures):
    # number of SNNs
    n = 15

    # data
    os.chdir("..")
    os.chdir("..")
    os.chdir("Patients")

    sph = 10
    window_length = 5

    os.chdir("pat_" + str(patient) + "_features")

    # load the vigilance data obtained from the model developed in:
    # https://hdl.handle.net/10316/97971
    vigilance = np.load("pat_" + str(patient) + "_vigilance", allow_pickle=True)
    vigilance_datetimes = np.load("pat_" + str(patient) + "_datetimes", allow_pickle=True)

    for i in range(len(vigilance)):
        vigilance[i] = np.abs(vigilance[i] - 1)
        vigilance[i] = np.clip(vigilance[i], 0.05, 0.95)

    # seizure onsets
    seizure_information = np.load("all_seizure_information.pkl", allow_pickle=True)

    ####################### Loading Testing Seizures #############################

    testing_data = []
    testing_labels = []
    testing_datetimes = []
    testing_onsets = []

    for seizure_k in range(3, total_seizures):
        # testing seizures
        seizure_data = np.load("pat_" + str(patient) + "_seizure_" + str(seizure_k) + "_features.npy")
        seizure_datetime = np.load("feature_datetimes_" + str(seizure_k) + ".npy")
        seizure_onset = float(seizure_information[seizure_k][0])

        # removing SPH
        [seizure_data, seizure_datetime] = utils.removeSPHfromSignal(seizure_data, seizure_datetime, seizure_onset)

        seizure_labels = utils.getLabelsForSeizure(seizure_datetime, sop, seizure_onset)

        seizure_labels = np.transpose(seizure_labels)

        # reshape the data
        seizure_data = seizure_data.reshape((seizure_data.shape[0], -1))

        testing_data.append(seizure_data)
        testing_labels.append(seizure_labels)
        testing_datetimes.append(seizure_datetime)
        testing_onsets.append(seizure_onset)

    os.chdir('..')
    os.chdir('..')
    os.chdir('Code')
    os.chdir('SNNs_ensemble_model')

    classification_labels_each_classifier = []
    for nn in range(0, n):
        features_input_layer = Input(shape=(1121,))

        x = Dropout(0.5)(features_input_layer)

        x = Dense(2)(x)

        output_layer = Activation('softmax')(x)

        model = Model(features_input_layer, output_layer)

        model.load_weights('Results/Patient ' + str(patient) + '/seizure_model_' + str(nn) + '.h5')

        norm_values = np.load('Results/Patient ' + str(patient) + '/norm_values_' + str(nn) + '.npy')

        #################### Classification ###########################
        classification_labels = []

        for i in range(0, len(testing_labels)):
            current_testing_data = (testing_data[i] - norm_values[0]) / norm_values[1]
            y_pred = model.predict(current_testing_data)
            y_pred = np.argmax(y_pred, axis=1)
            classification_labels.append(y_pred)

        classification_labels_each_classifier.append(classification_labels)

    # voting system
    number_of_tested_seizures = len(classification_labels_each_classifier[0])
    number_of_classifiers = len(classification_labels_each_classifier)
    for i in range(0, number_of_tested_seizures):
        voted_labels = np.zeros(len(classification_labels_each_classifier[0][i]))
        for j in range(0, number_of_classifiers):
            voted_labels = voted_labels + classification_labels_each_classifier[j][i]

        voted_labels = voted_labels / number_of_classifiers
        voted_labels = np.where(voted_labels > 0.5, 1, 0)

        classification_labels[i] = voted_labels

    predicted_labels = classification_labels.copy()
    forecast_labels = classification_labels.copy()
    exact_labels = classification_labels.copy()

    ###################### Postprocessing ######################
    for i in range(len(testing_labels)):
        predicted_labels[i] = utils.FiringPowerAndRefractoryPeriod(classification_labels[i], testing_datetimes[i], sop,
                                                                   sph, window_length)
        forecast_labels[i], exact_labels[i] = utils.FiringPower(classification_labels[i], sop, window_length, "fore")

    return [testing_datetimes, testing_labels, predicted_labels, exact_labels, vigilance, vigilance_datetimes]
