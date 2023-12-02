import datetime
from datetime import timedelta
import numpy as np
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.utils import class_weight
from scipy import stats
from sklearn import metrics


def convertIntoDatetime(date):
    return datetime.datetime.fromtimestamp(date)


def removeSPHfromSignal(seizure_data, seizure_datetime, seizure_onset):
    seizure_onset = convertIntoDatetime(seizure_onset)
    sph_datetime = seizure_onset - timedelta(minutes=10)

    final_index = 0
    for i in range(len(seizure_datetime) - 1, 1, -1):
        current_datetime = datetime.datetime.fromtimestamp(seizure_datetime[i])
        if not (current_datetime > sph_datetime):
            final_index = i
            break
    seizure_datetime = seizure_datetime[0:final_index]
    seizure_data = seizure_data[0:final_index, :, :]

    return seizure_data, seizure_datetime


def getLabelsForSeizure(seizure_datetime, sop, seizure_onset):
    seizure_onset = convertIntoDatetime(seizure_onset)
    preictal_datetime = seizure_onset - timedelta(minutes=(10 + sop))

    final_index = 0
    for i in range(len(seizure_datetime) - 1, 1, -1):
        current_datetime = datetime.datetime.fromtimestamp(seizure_datetime[i])
        if not (current_datetime > preictal_datetime):
            final_index = i
            break
    labels = np.zeros(len(seizure_datetime))
    labels[final_index:] = 1

    return labels


def removeConstantFeatures(features):
    constant_features_index = []

    for i in range(features.shape[1]):
        if np.var(features[:, i]) < 1e-9:
            constant_features_index.append(i)

    features = np.delete(features, constant_features_index, axis=1)
    return [constant_features_index, features]


def removeRedundantFeatures(features):
    redundant_features_index = []

    for i in range(features.shape[1]):
        for j in range(i, features.shape[1]):
            if i != j and abs(np.corrcoef(features[:, i], features[:, j])[0][1]) > 0.95:
                redundant_features_index.append(j)

    features = np.delete(features, redundant_features_index, axis=1)
    return [redundant_features_index, features]


def computeSampleWeights(labels, class_weights):
    sample_weights = np.zeros(len(labels))

    sample_weights[np.where(labels == 0)[0]] = class_weights[0]
    sample_weights[np.where(labels == 1)[0]] = class_weights[1]

    return sample_weights


def computeBalancedClassWeights(labels):
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(labels),
                                                      y=labels)
    return class_weights


def specificity(tn, fp):
    return tn / (tn + fp)


def sensitivity(tp, fn):
    return tp / (tp + fn)


def systematic_random_undersampling(target):
    idx_class0 = np.where(target == 0)[0]
    idx_class1 = np.where(target == 1)[0]
    if len(idx_class1) >= len(idx_class0):
        idx_majority_class = idx_class1
        idx_minority_class = idx_class0
    elif len(idx_class1) < len(idx_class0):
        idx_majority_class = idx_class0
        idx_minority_class = idx_class1

    n_groups = len(idx_minority_class)
    n_samples = len(idx_majority_class)
    min_samples = n_samples // n_groups
    remaining_samples = n_samples % n_groups
    n_samples_per_group = [min_samples + 1] * remaining_samples + [min_samples] * (n_groups - remaining_samples)

    idx_selected = []
    begin_idx = 0
    for i in n_samples_per_group:
        end_idx = begin_idx + i

        idx_group = idx_majority_class[begin_idx:end_idx]
        idx = np.random.choice(idx_group)
        idx_selected.append(idx)

        begin_idx = end_idx

    [idx_selected.append(idx) for idx in idx_minority_class]

    idx_selected = np.sort(idx_selected)

    return idx_selected


def FiringPowerAndRefractoryPeriod(predicted_labels, datetimes, sop, sph, window_length):
    predicted_labels, exact_labels = FiringPower(predicted_labels, sop, window_length, "pred")
    predicted_labels = RefractoryPeriod(predicted_labels, datetimes, sop, sph)

    return predicted_labels


def FiringPower(classification_labels, sop, window_length, c_type):
    kernel_size = int(sop * (60 / window_length))
    kernel = np.ones(kernel_size) / kernel_size
    classification_labels = np.convolve(classification_labels, kernel, mode='same')

    for i in range(len(classification_labels)):
        if classification_labels[i] > 1:
            classification_labels[i] = 1

    exact_labels = classification_labels[:]

    if c_type == "pred":
        threshold = 0.7
        classification_labels = [1 if classification_labels_ > threshold else 0 for classification_labels_ in
                                 classification_labels]
    elif c_type == "fore":
        classification_labels = [
            1 if classification_labels_ > 0.7 else 2 if (0.3 < classification_labels_ <= 0.7) else 3 for
            classification_labels_ in classification_labels]

    return classification_labels, exact_labels


def RefractoryPeriod(predicted_labels, datetimes, sop, sph):
    refractory_on = False
    for i in range(len(predicted_labels)):
        if not refractory_on:
            if predicted_labels[i] == 1:
                refractory_on = True
                onset_alarm = convertIntoDatetime(datetimes[i])
        else:
            predicted_labels[i] = 0

            if convertIntoDatetime(datetimes[i]) > (onset_alarm + timedelta(minutes=sop + sph)):
                refractory_on = False

    return predicted_labels


def didItPredictTheSeizure(classified, labels):
    preictal_length = len(np.argwhere(labels))
    preictal_beginning = np.argwhere(labels)[0][0]

    did_it_predict_the_seizure = 1 in classified[preictal_beginning:preictal_beginning + preictal_length]

    return did_it_predict_the_seizure


def seizureSensitivity(classified, labels):
    seizure_sensitivity = 0
    for i in range(len(classified)):
        seizure_sensitivity += didItPredictTheSeizure(classified[i], labels[i])
    seizure_sensitivity = seizure_sensitivity / len(classified)

    return seizure_sensitivity


def falsePositiveRate(predicted, labels, preictal, datetime, onsets, window_length):
    false_alarms = 0
    interictal = 0
    refractory = 0

    for i in range(len(predicted)):
        preictal_beginning = np.argwhere(labels[i])[0][0]
        interictal_ending = convertIntoDatetime(datetime[i][preictal_beginning]) - timedelta(seconds=window_length)
        interictal_beginning = convertIntoDatetime(datetime[i][0])
        interictal_duration = interictal_ending - interictal_beginning
        interictal_length = interictal_duration.total_seconds() / 3600
        number_false_alarms = np.sum(predicted[i][0:preictal_beginning])
        false_alarm_indexes = np.argwhere(predicted[i][0:preictal_beginning])

        onset = convertIntoDatetime(onsets[i])
        preictal_onset = onset - timedelta(minutes=preictal)

        ref_time = 0
        for j in range(number_false_alarms):
            time_false_alarm = convertIntoDatetime(datetime[i][false_alarm_indexes[j][0]])
            time_to_preictal = preictal_onset - time_false_alarm
            if time_to_preictal.total_seconds() > (preictal * 60):
                ref_time += preictal / 60
            else:
                ref_time += time_to_preictal.total_seconds() / 3600

        false_alarms += number_false_alarms
        interictal += interictal_length
        refractory += ref_time

    FPR = false_alarms / (interictal - refractory)

    return FPR


def timeInWarning(forecast):
    time_in_warning = 0
    for i in range(len(forecast)):
        forecast[i] = np.array(forecast[i])
        warning = np.argwhere(forecast[i] < 2)
        time_in_warning += len(warning) / len(forecast[i])
    time_in_warning = time_in_warning / len(forecast)

    return time_in_warning


def brierScore(forecast, labels):
    brier_score = 0
    for i in range(len(forecast)):
        brier_score += metrics.brier_score_loss(labels[i], forecast[i])

    brier_score = brier_score / len(forecast)

    return brier_score


def surrogateScore(forecast, labels):

    forecast_full = np.concatenate(forecast)
    labels_full = np.concatenate(labels)

    n_surrogates = 1000
    surrogate_brier_scores = []

    for i in range(n_surrogates):
        surrogate_forecast = np.random.permutation(forecast_full)
        brier_score_surrogate = metrics.brier_score_loss(labels_full, surrogate_forecast)
        surrogate_brier_scores.append(brier_score_surrogate)

    return surrogate_brier_scores


def brierSkillScore(brier_score, forecast, labels):

    reference_scores = surrogateScore(forecast, labels)

    reference_score = np.mean(reference_scores)

    brier_skill_score = 1 - (brier_score / reference_score)

    return brier_skill_score


def reliabilityCurve(forecast, labels, patient):
    forecast_full = np.concatenate(forecast)
    labels_full = np.concatenate(labels)

    prob_true, prob_fore = calibration_curve(labels_full, forecast_full, n_bins=10)

    plt.plot(prob_fore, prob_true, marker='o', linewidth=1)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Forecast probability')
    plt.ylabel('Observed frequency')

    plt.savefig("RL-" + str(patient) + ".pdf", dpi=200, bbox_inches='tight')
    plt.savefig("RL-" + str(patient) + ".png", dpi=200, bbox_inches='tight')
    # plt.show()
    plt.clf()


def surrogateSensitivity(predicted, datetimes, onset, sop, sph):
    seizure_sensitivity = 0
    surrogate_labels = shuffle_labels(datetimes, onset, sop, sph)
    seizure_sensitivity = seizure_sensitivity + didItPredictTheSeizure(predicted, surrogate_labels)

    return seizure_sensitivity


def shuffle_labels(datetimes, onset, sop, sph):
    firing_power_threshold = 0.7
    surrogate_labels = np.zeros(len(datetimes))

    pre_ictal_beginning = convertIntoDatetime(onset)
    pre_ictal_beginning = pre_ictal_beginning - timedelta(minutes=sop + sph)

    minimum_index = sop * firing_power_threshold
    maximum_index = 0
    for i in range(len(datetimes) - 1, 0, -1):
        current_moment = convertIntoDatetime(datetimes[i])
        if pre_ictal_beginning > current_moment:
            maximum_index = i
            break

    sop_beginning_index = np.random.randint(minimum_index, maximum_index)

    end_time = convertIntoDatetime(datetimes[sop_beginning_index])
    end_time = end_time + timedelta(minutes=sop)

    for i in range(sop_beginning_index, len(datetimes)):
        current_moment = convertIntoDatetime(datetimes[i])
        if current_moment > end_time:
            break
        else:
            surrogate_labels[i] = 1

    return surrogate_labels


def t_test_one_independent_mean(population_mean, population_std, sample_mean, number_samples):
    tt = abs(population_mean - sample_mean) / (population_std / np.sqrt(number_samples))

    pval = stats.t.sf(np.abs(tt), number_samples - 1) * 2

    return [tt, pval]


def statisticalValidation(classification, labels, seizure_sensitivity, sop, sph, testing_datetimes, testing_onsets):
    surrogate_sensitivity = []
    for i in range(0, len(labels)):
        for j in range(0, 30):
            surrogate_sensitivity.append(
                surrogateSensitivity(classification[i], testing_datetimes[i],
                                     testing_onsets[i], sop, sph))

    surrogate_sensitivity = surrogate_sensitivity
    print("Surrogate Sensitivity " + str(np.mean(surrogate_sensitivity)) + " +/- " + str(np.std(surrogate_sensitivity)))

    val = 0
    pval = 1
    print("Does it perform above chance?")
    if np.mean(surrogate_sensitivity) < seizure_sensitivity:
        [tt, pval] = t_test_one_independent_mean(np.mean(surrogate_sensitivity), np.std(surrogate_sensitivity),
                                                 seizure_sensitivity, 30)
        if pval < 0.05:
            print("Yes")
            val = 1
        else:
            print("No")
    else:
        print("No")

    return [np.mean(surrogate_sensitivity), np.std(surrogate_sensitivity), pval, val]


def statisticalValidationBS(forecast, labels, brier_score):

    surrogate_score = surrogateScore(forecast, labels)
    print("Surrogate Score " + str(np.mean(surrogate_score)) + " +/- " + str(np.std(surrogate_score)))

    val = 0
    pval = 1
    print("Does it perform above chance?")
    if np.mean(surrogate_score) > brier_score:
        [tt, pval] = t_test_one_independent_mean(np.mean(surrogate_score), np.std(surrogate_score),
                                                 brier_score, 1000)
        if pval < 0.05:
            print("Yes")
            val = 1
        else:
            print("No")
    else:
        print("No")

    return [np.mean(surrogate_score), np.std(surrogate_score), pval, val]
