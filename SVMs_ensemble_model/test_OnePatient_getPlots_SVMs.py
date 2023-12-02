import os
import numpy as np
import utils

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

# due to the np.delete function
import warnings

warnings.filterwarnings("ignore")


def testOnePatientGetPlots(patient, sop, number_features, c_value, total_seizures):
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

    # training seizures
    seizure_1_data = np.load("pat_" + str(patient) + "_seizure_0_features.npy")
    seizure_2_data = np.load("pat_" + str(patient) + "_seizure_1_features.npy")
    seizure_3_data = np.load("pat_" + str(patient) + "_seizure_2_features.npy")

    # seizure datetimes
    seizure_1_datetime = np.load("feature_datetimes_0.npy")
    seizure_2_datetime = np.load("feature_datetimes_1.npy")
    seizure_3_datetime = np.load("feature_datetimes_2.npy")

    # seizure onsets
    seizure_information = np.load("all_seizure_information.pkl", allow_pickle=True)
    seizure_onset_1 = float(seizure_information[0][0])
    seizure_onset_2 = float(seizure_information[1][0])
    seizure_onset_3 = float(seizure_information[2][0])

    # removing sph
    [seizure_1_data, seizure_1_datetime] = utils.removeSPHfromSignal(seizure_1_data, seizure_1_datetime,
                                                                     seizure_onset_1)
    [seizure_2_data, seizure_2_datetime] = utils.removeSPHfromSignal(seizure_2_data, seizure_2_datetime,
                                                                     seizure_onset_2)
    [seizure_3_data, seizure_3_datetime] = utils.removeSPHfromSignal(seizure_3_data, seizure_3_datetime,
                                                                     seizure_onset_3)

    seizure_1_labels = utils.getLabelsForSeizure(seizure_1_datetime, sop, seizure_onset_1)
    seizure_2_labels = utils.getLabelsForSeizure(seizure_2_datetime, sop, seizure_onset_2)
    seizure_3_labels = utils.getLabelsForSeizure(seizure_3_datetime, sop, seizure_onset_3)

    training_features = np.concatenate([seizure_1_data, seizure_2_data, seizure_3_data], axis=0)
    training_labels = np.concatenate([seizure_1_labels, seizure_2_labels, seizure_3_labels], axis=0)

    # reshape the training feature vector
    training_features = np.reshape(training_features, (training_features.shape[0],
                                                       training_features.shape[1] * training_features.shape[2]))

    del seizure_1_data
    del seizure_2_data
    del seizure_3_data

    del seizure_1_labels
    del seizure_2_labels
    del seizure_3_labels

    del seizure_1_datetime
    del seizure_2_datetime
    del seizure_3_datetime

    del seizure_onset_1
    del seizure_onset_2
    del seizure_onset_3

    ####################### Loading Testing Seizures #############################

    testing_features = []
    testing_labels = []
    testing_datetimes = []
    testing_onsets = []
    for seizure_k in range(3, total_seizures):
        # testing seizures
        seizure_features = np.load("pat_" + str(patient) + "_seizure_" + str(seizure_k) + "_features.npy")
        seizure_datetime = np.load("feature_datetimes_" + str(seizure_k) + ".npy")
        seizure_onset = float(seizure_information[seizure_k][0])

        # removing SPH
        [seizure_features, seizure_datetime] = utils.removeSPHfromSignal(seizure_features, seizure_datetime,
                                                                         seizure_onset)

        seizure_labels = utils.getLabelsForSeizure(seizure_datetime, sop, seizure_onset)

        # reshape the feature vector
        seizure_features = np.reshape(seizure_features, (seizure_features.shape[0],
                                                         seizure_features.shape[1] * seizure_features.shape[2]))

        seizure_labels = np.transpose(seizure_labels)

        testing_features.append(seizure_features)
        testing_labels.append(seizure_labels)
        testing_datetimes.append(seizure_datetime)
        testing_onsets.append(seizure_onset)

    ################### Missing value imputation ##################################
    # training
    missing_values_indexes = np.unique(np.argwhere(np.isnan(training_features))[:, 0])
    training_features = np.delete(training_features, missing_values_indexes, axis=0)
    training_labels = np.delete(training_labels, missing_values_indexes, axis=0)
    # testing
    for i in range(len(testing_labels)):
        missing_values_indexes = np.unique(np.argwhere(np.isnan(testing_features[i]))[:, 0])

        testing_features[i] = np.delete(testing_features[i], missing_values_indexes, axis=0)
        testing_labels[i] = np.delete(testing_labels[i], missing_values_indexes, axis=0)
        testing_datetimes[i] = np.delete(testing_datetimes[i], missing_values_indexes, axis=0)

    ################## Removing Constant Values ###########
    # training
    [constant_indexes, training_features] = utils.removeConstantFeatures(training_features)
    # testing
    for i in range(len(testing_labels)):
        testing_features[i] = np.delete(testing_features[i], constant_indexes, axis=1)

    #################### Standardization #######################
    # training features
    scaler = StandardScaler().fit(training_features)
    training_features = scaler.transform(training_features)
    # testing features
    for i in range(len(testing_labels)):
        testing_features[i] = scaler.transform(testing_features[i])

    classification_labels_each_classifier = []
    # train 15 classifiers
    for iii in range(15):

        testing_features_j = testing_features.copy()

        #################### Data Balancing - Sampling ###########################
        # random undersampling
        idx_selected = utils.systematic_random_undersampling(training_labels)
        training_features_i = training_features[idx_selected, :]
        training_labels_i = training_labels[idx_selected]

        #################### Feature Selection #######################
        n_features = number_features
        rf = RandomForestClassifier(max_depth=10, random_state=42, n_estimators=100).fit(training_features_i,
                                                                                         training_labels_i)
        feature_importance_indexes = np.argsort(((-1) * rf.feature_importances_))
        selected_features_indexes = feature_importance_indexes[0:n_features]
        # training features
        training_features_i = training_features_i[:, selected_features_indexes]
        # testing features
        for i in range(len(testing_labels)):
            testing_features_j[i] = testing_features_j[i][:, selected_features_indexes]

        #################### Classification ###########################
        classification_labels = []

        svm_model = svm.LinearSVC(C=c_value, dual=False)
        svm_model.fit(training_features_i, training_labels_i)

        for i in range(0, len(testing_labels)):
            classification_labels.append(svm_model.predict(testing_features_j[i]))

        classification_labels_each_classifier.append(classification_labels)

    # voting system
    number_of_tested_seizures = len(classification_labels_each_classifier[0])
    number_of_classifiers = len(classification_labels_each_classifier)
    for i in range(number_of_tested_seizures):
        voted_labels = np.zeros(len(classification_labels_each_classifier[0][i]))
        for j in range(number_of_classifiers):
            voted_labels = voted_labels + classification_labels_each_classifier[j][i]

        voted_labels = voted_labels / number_of_classifiers
        voted_labels = np.where(voted_labels > 0.5, 1, 0)

        classification_labels[i] = voted_labels

    predicted_labels = classification_labels.copy()
    forecast_labels = classification_labels.copy()
    exact_labels = classification_labels.copy()

    ###################### Postprocessing ######################
    for i in range(len(testing_labels)):
        predicted_labels[i] = utils.FiringPowerAndRefractoryPeriod(predicted_labels[i], testing_datetimes[i], sop, sph,
                                                                   window_length)
        forecast_labels[i], exact_labels[i] = utils.FiringPower(classification_labels[i], sop, window_length, "fore")

    os.chdir('..')
    os.chdir('..')
    os.chdir('Code')
    os.chdir('SVMs_ensemble_model')

    return [testing_datetimes, testing_labels, predicted_labels, exact_labels, vigilance, vigilance_datetimes]
