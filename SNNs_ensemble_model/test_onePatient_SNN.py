import os
import numpy as np
import utils
import tensorflow as tf

from keras.models import Model
from keras.layers import *

# due to the np.delete function
import warnings

warnings.filterwarnings("ignore")


def testOnePatient(patient, sop, total_seizures):
    # number of SNNs
    n = 15

    # data
    os.chdir("..")
    os.chdir("..")
    os.chdir("Patients")

    sph = 10
    window_length = 5

    os.chdir("pat_" + str(patient) + "_features")

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

    ##################### Performance Metrics/Statistical Validation #####################
    print("\n############ Seizure Prediction ############")
    seizure_sensitivityP = utils.seizureSensitivity(predicted_labels, testing_labels)
    FPR = utils.falsePositiveRate(predicted_labels, testing_labels, sph + sop, testing_datetimes, testing_onsets,
                                  window_length)

    print("Sensitivity: " + str(seizure_sensitivityP))
    print("FPR/h: " + str(FPR))

    print("############ Validation ############")
    [surrogate_ssP, surrogate_stdP, pvalP, valP] = utils.statisticalValidation(predicted_labels, testing_labels,
                                                                               seizure_sensitivityP, sop, sph,
                                                                               testing_datetimes, testing_onsets)

    print("############ Seizure Forecasting ############")
    seizure_sensitivityF = utils.seizureSensitivity(forecast_labels, testing_labels)
    time_in_warning = utils.timeInWarning(forecast_labels)
    brier_score = utils.brierScore(exact_labels, testing_labels)
    brier_skill_score = utils.brierSkillScore(brier_score, exact_labels, testing_labels)

    print("Sensitivity: " + str(seizure_sensitivityF))
    print("Time in Warning: " + str(time_in_warning))
    print("Brier Score: " + str(brier_score))
    print("Brier Skill Score: " + str(brier_skill_score))

    print("############ Validation ############")
    [surrogate_ssF1, surrogate_stdF1, pvalF1, valF1] = utils.statisticalValidation(forecast_labels, testing_labels,
                                                                                   seizure_sensitivityF, sop, sph,
                                                                                   testing_datetimes, testing_onsets)

    [surrogate_ssF2, surrogate_stdF2, pvalF2, valF2] = utils.statisticalValidationBS(exact_labels, testing_labels,
                                                                                     brier_score)

    os.chdir('Results')
    utils.reliabilityCurve(exact_labels, testing_labels, patient)
    os.chdir('..')

    return [patient, seizure_sensitivityP, FPR, surrogate_ssP, surrogate_stdP, pvalP, valP, seizure_sensitivityF,
            time_in_warning, brier_score, brier_skill_score, valF1, valF2]
