import os
import numpy as np
import utils

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import *
import tensorflow as tf
from keras.utils import to_categorical

from trainSNN import train_SNN
from trainSNN import train_SNN_and_save

# due to the np.delete function
import warnings

warnings.filterwarnings("ignore")


def train_model_SNN(patient):
    # data
    os.chdir("..")
    os.chdir("..")
    os.chdir("Patients")

    os.chdir("pat_" + str(patient) + "_features")

    # runs for each preictal
    k = 5
    # final models
    n = 15

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

    # removing  sph
    [seizure_1_data, seizure_1_datetime] = utils.removeSPHfromSignal(seizure_1_data, seizure_1_datetime,
                                                                     seizure_onset_1)
    [seizure_2_data, seizure_2_datetime] = utils.removeSPHfromSignal(seizure_2_data, seizure_2_datetime,
                                                                     seizure_onset_2)
    [seizure_3_data, seizure_3_datetime] = utils.removeSPHfromSignal(seizure_3_data, seizure_3_datetime,
                                                                     seizure_onset_3)

    training_data = np.concatenate([seizure_1_data, seizure_2_data, seizure_3_data], axis=0)

    del seizure_1_data
    del seizure_2_data
    del seizure_3_data

    # make variables to store the grid-search performance
    sop_gridsearch_values = [20, 25, 30, 35, 40, 45, 50]
    performance_values = np.zeros([len(sop_gridsearch_values)])

    for i in range(len(sop_gridsearch_values)):
        for kk in range(k):
            seizure_1_labels = utils.getLabelsForSeizure(seizure_1_datetime, sop_gridsearch_values[i], seizure_onset_1)
            seizure_2_labels = utils.getLabelsForSeizure(seizure_2_datetime, sop_gridsearch_values[i], seizure_onset_2)
            seizure_3_labels = utils.getLabelsForSeizure(seizure_3_datetime, sop_gridsearch_values[i], seizure_onset_3)

            training_labels = np.concatenate([seizure_1_labels, seizure_2_labels, seizure_3_labels], axis=0)

            training_data_i, validation_data, training_labels, validation_labels = train_test_split(training_data,
                                                                                                    training_labels,
                                                                                                    test_size=0.2,
                                                                                                    random_state=0,
                                                                                                    shuffle=True,
                                                                                                    stratify=
                                                                                                    training_labels)

            #################### Data Balancing - Sampling ###########################
            # random undersampling
            idx_selected = utils.systematic_random_undersampling(training_labels)
            training_data_i = training_data_i[idx_selected, :]
            training_labels = training_labels[idx_selected]

            # reshape data
            training_data_i = training_data_i.reshape((training_data_i.shape[0], -1))
            validation_data = validation_data.reshape((validation_data.shape[0], -1))

            [model, validation_data, validation_labels, norm_values] = train_SNN(training_data_i, training_labels,
                                                                                 validation_data, validation_labels)

            ###################### Classification #######################
            predicted_labels = model.predict(validation_data)
            predicted_labels = np.argmax(predicted_labels, axis=1)
            validation_labels = np.argmax(validation_labels, axis=1)

            tn, fp, fn, tp = confusion_matrix(validation_labels, predicted_labels).ravel()

            performance = np.sqrt(utils.specificity(tn, fp) * utils.sensitivity(tp, fn))

            performance_values[i] = performance_values[i] + performance

    index_best_models = np.argmax(performance_values)
    best_sop = sop_gridsearch_values[index_best_models]

    os.chdir("..")
    os.chdir("..")
    os.chdir("Code")
    os.chdir("SNNs_ensemble_model")

    # training final models
    for nn in range(n):
        seizure_1_labels = utils.getLabelsForSeizure(seizure_1_datetime, best_sop, seizure_onset_1)
        seizure_2_labels = utils.getLabelsForSeizure(seizure_2_datetime, best_sop, seizure_onset_2)
        seizure_3_labels = utils.getLabelsForSeizure(seizure_3_datetime, best_sop, seizure_onset_3)

        training_labels = np.concatenate([seizure_1_labels, seizure_2_labels, seizure_3_labels], axis=0)

        training_data_i, validation_data, training_labels, validation_labels = train_test_split(training_data,
                                                                                                training_labels,
                                                                                                test_size=0.2,
                                                                                                random_state=0,
                                                                                                shuffle=True,
                                                                                                stratify=
                                                                                                training_labels)

        #################### Data Balancing - Sampling ###########################
        # random undersampling
        idx_selected = utils.systematic_random_undersampling(training_labels)
        training_data_i = training_data_i[idx_selected, :]
        training_labels = training_labels[idx_selected]

        # reshape data
        training_data_i = training_data_i.reshape((training_data_i.shape[0], -1))
        validation_data = validation_data.reshape((validation_data.shape[0], -1))

        train_SNN_and_save(training_data_i, training_labels, validation_data, validation_labels, patient, nn)

    return [patient, best_sop]
