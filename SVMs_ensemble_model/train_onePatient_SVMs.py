import os
import numpy as np
import utils
import time

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

# due to the np.delete function
import warnings

warnings.filterwarnings("ignore")

# data
os.chdir("..")
os.chdir("..")
os.chdir("Patients")


def calculatePreIctalAndFeatureNumber(patient):
    os.chdir("pat_" + str(patient) + "_features")

    t = time.process_time()
    contador = 0

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

    # make variables to store the grid-search performance
    sop_gridsearch_values = [20, 25, 30, 35, 40, 45, 50]
    k_features = [3, 5, 7, 10, 15, 20, 30]
    c_pot = np.arange(start=-10, stop=10, step=2, dtype=float)
    C = 2 ** c_pot
    performance_values = np.zeros([len(sop_gridsearch_values), len(k_features), len(C)])

    for i in range(len(sop_gridsearch_values)):
        seizure_1_labels = utils.getLabelsForSeizure(seizure_1_datetime, sop_gridsearch_values[i], seizure_onset_1)
        seizure_2_labels = utils.getLabelsForSeizure(seizure_2_datetime, sop_gridsearch_values[i], seizure_onset_2)
        seizure_3_labels = utils.getLabelsForSeizure(seizure_3_datetime, sop_gridsearch_values[i], seizure_onset_3)
        for kk in range(len(k_features)):
            for c in range(len(C)):
                c_value = C[c]
                for k in range(3):
                    # seizure_1 for validation, seizure_2 and seizure_3 for training
                    if k == 0:
                        validation_features = seizure_1_data
                        validation_labels = seizure_1_labels

                        training_features_1 = seizure_2_data
                        training_labels_1 = seizure_2_labels

                        training_features_2 = seizure_3_data
                        training_labels_2 = seizure_3_labels

                    # seizure_2 for validation, seizure_1 and seizure_3 for training
                    elif k == 1:
                        validation_features = seizure_2_data
                        validation_labels = seizure_2_labels

                        training_features_1 = seizure_1_data
                        training_labels_1 = seizure_1_labels

                        training_features_2 = seizure_3_data
                        training_labels_2 = seizure_3_labels

                    # seizure_3 for validation, seizure_1 and seizure_2 for training
                    elif k == 2:
                        validation_features = seizure_3_data
                        validation_labels = seizure_3_labels

                        training_features_1 = seizure_1_data
                        training_labels_1 = seizure_1_labels

                        training_features_2 = seizure_2_data
                        training_labels_2 = seizure_2_labels

                    training_features = np.concatenate([training_features_1, training_features_2], axis=0)
                    training_labels = np.concatenate([training_labels_1, training_labels_2], axis=0)

                    training_features = np.reshape(training_features,
                                                   (training_features.shape[0],
                                                    training_features.shape[1] * training_features.shape[2]))

                    validation_features = np.reshape(validation_features,
                                                     (validation_features.shape[0],
                                                      validation_features.shape[1] * validation_features.shape[2]))

                    del training_features_1
                    del training_features_2
                    del training_labels_1
                    del training_labels_2


                    ################### Missing value imputation ###############
                    missing_values_indexes = np.unique(np.argwhere(np.isnan(training_features))[:, 0])
                    training_features = np.delete(training_features, missing_values_indexes, axis=0)
                    training_labels = np.delete(training_labels, missing_values_indexes, axis=0)

                    missing_values_indexes = np.unique(np.argwhere(np.isnan(validation_features))[:, 0])
                    validation_features = np.delete(validation_features, missing_values_indexes, axis=0)
                    validation_labels = np.delete(validation_labels, missing_values_indexes, axis=0)

                    ################## Removing Constant and Redundant Values ###########
                    [constant_indexes, training_features] = utils.removeConstantFeatures(training_features)
                    validation_features = np.delete(validation_features, constant_indexes, axis=1)

                    #################### Standardization #######################
                    scaler = StandardScaler().fit(training_features)
                    training_features = scaler.transform(training_features)
                    validation_features = scaler.transform(validation_features)

                    # train 15 classifiers
                    for iii in range(15):
                        #################### Data Balancing - Sampling ###########################
                        # random undersampling
                        idx_selected = utils.systematic_random_undersampling(training_labels)
                        training_features_i = training_features[idx_selected, :]
                        training_labels_i = training_labels[idx_selected]

                        #################### Feature Selection #######################
                        n_features = k_features[kk]

                        rf = RandomForestClassifier(max_depth=10, random_state=42, n_estimators=100).fit(
                            training_features_i, training_labels_i)

                        feature_importance_indexes = np.argsort(((-1) * rf.feature_importances_))
                        selected_features_indexes = feature_importance_indexes[0:n_features]

                        training_features_i = training_features_i[:, selected_features_indexes]
                        validation_features_i = validation_features[:, selected_features_indexes]

                        #################### Classification ###########################
                        svm_model = svm.LinearSVC(C=c_value, dual=False)
                        svm_model.fit(training_features_i, training_labels_i)

                        ###################### Performance Evaluation #########################
                        predicted_labels = svm_model.predict(validation_features_i)
                        tn, fp, fn, tp = confusion_matrix(validation_labels, predicted_labels).ravel()

                        performance = np.sqrt(utils.specificity(tn, fp) * utils.sensitivity(tp, fn))

                        performance_values[i, kk, c] = performance_values[i, kk, c] + performance

                    contador = contador + 1
                    if contador % 50 == 0 or contador == 1 or contador == 7 * 7 * 3 * 10:
                        print(str(contador) + " of " + str(7 * 7 * 3 * 10) + " iterations")

    elapsed_time = time.process_time() - t

    print("Pre-ictal search finished successfully")
    print("Elapsed Time: " + str(elapsed_time))

    # dividing by 3, to have a normalized (0-1) performance value
    performance_values = performance_values / (3 * 30)

    # best set of sop and features
    best_set = np.unravel_index(performance_values.argmax(), performance_values.shape)
    chosen_pre_ictal = sop_gridsearch_values[best_set[0]]
    chosen_k_features = k_features[best_set[1]]
    chosen_c_value = C[best_set[2]]

    print("Patient " + str(patient) + ", Pre-Ictal: " + str(chosen_pre_ictal) + " min with " + str(
        chosen_k_features) + " features and C-value: " + str(chosen_c_value))

    os.chdir('..')

    return [patient, chosen_pre_ictal, chosen_k_features, chosen_c_value]
