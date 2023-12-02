from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import *
import tensorflow as tf
from keras.utils import to_categorical
import os
import numpy as np


def train_SNN(training_data_i, training_labels, validation_data, validation_labels):

    features_input_layer = Input(shape=(1211,))

    x = Dropout(0.5)(features_input_layer)

    x = Dense(2)(x)

    output_layer = Activation('softmax')(x)

    model = Model(features_input_layer, output_layer)

    model.compile(optimizer=Adam(learning_rate=3e-4), loss='binary_crossentropy', metrics='acc')
    model.summary()

    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=50)

    callbacks_parameters = [early_stopping_cb]
    training_labels = to_categorical(training_labels, 2)
    validation_labels = to_categorical(validation_labels, 2)

    norm_values = [np.mean(training_data_i), np.std(training_data_i)]
    training_data_i = (training_data_i - norm_values[0]) / norm_values[1]
    validation_data = (validation_data - norm_values[0]) / norm_values[1]

    model.fit(training_data_i, training_labels, epochs=500,
              verbose=1, validation_data=(validation_data, validation_labels),
              callbacks=callbacks_parameters)

    return [model, validation_data, validation_labels, norm_values]


def train_SNN_and_save(training_data_i, training_labels, validation_data, validation_labels, patient, nn):

    features_input_layer = Input(shape=(1121,))

    x = Dropout(0.5)(features_input_layer)

    x = Dense(2)(x)

    output_layer = Activation('softmax')(x)

    model = Model(features_input_layer, output_layer)

    model.compile(optimizer=Adam(learning_rate=3e-4), loss='binary_crossentropy', metrics='acc')
    model.summary()

    if not os.path.isdir('Results\Patient ' + str(patient)):
        os.mkdir('Results\Patient ' + str(patient))

    model_checkpoint_cb = ModelCheckpoint(
        'Results\Patient ' + str(patient) + '\seizure_model_' + str(nn) + '.h5', 'val_loss',
        save_best_only=True, verbose=1, mode='min')

    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=50)

    callbacks_parameters = [model_checkpoint_cb, early_stopping_cb]

    training_labels = to_categorical(training_labels, 2)
    validation_labels = to_categorical(validation_labels, 2)

    norm_values = [np.mean(training_data_i), np.std(training_data_i)]
    training_data_i = (training_data_i - norm_values[0]) / norm_values[1]
    validation_data = (validation_data - norm_values[0]) / norm_values[1]

    model.fit(training_data_i, training_labels, epochs=500,
              verbose=1, validation_data=(validation_data, validation_labels),
              callbacks=callbacks_parameters)

    np.save('Results/Patient ' + str(patient) + '/norm_values_' + str(nn) + '.npy', norm_values)

    return [model, validation_data, validation_labels, norm_values]
