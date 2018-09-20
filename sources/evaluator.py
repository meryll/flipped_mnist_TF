import numpy as np
import os
from sources.utils import plot_utils

BINS = [0, 1]
PREDICTIONS_PATH = '../generated_files/predictions/'
FALSE_POSITIVES_DIR = 'false_positives'
FALSE_NEGATIVES_DIR = 'false_negatives'

# todo use override estimator evaluator
def evaluate(dataset, predictions):
    tp, tn, fp, fn = _get_confusion_matrix(dataset=dataset, predictions=predictions)
    # show_histogram(tp, tn, fp, fn)
    _show_for_each_digit(tp, tn, fp, fn)


def _show_for_each_digit(tp, tn, fp, fn):
    digits = range(10)

    for current_digit in digits:

        tp_axis = [angle for digit, angle in tp if digit == current_digit]
        fp_axis = [angle for digit, angle in fp if digit == current_digit]
        tn_axis = [angle for digit, angle in tn if digit == current_digit]
        fn_axis = [angle for digit, angle in fn if digit == current_digit]

        all_angles = tp_axis + fp_axis + tn_axis + fn_axis
        accuracy = (len(tp_axis) + len(tn_axis)) / len(all_angles) * 100
        precision = (len(tp_axis))/(len(tp_axis) + len(fp_axis))
        recall = (len(tp_axis))/(len(tp_axis) + len(fn_axis))

        fn_freq = _get_frequency(fn_axis, bins=BINS)

        print("----------{}------------".format(current_digit))
        print("Digit {} was found {} times.".format(current_digit, len(all_angles)))
        print("Accuracy: {} | Recall: {} | Precision: {}".format(accuracy, recall, precision))
        print("TP: {} | TN: {}".format(len(tp_axis), len(tn_axis)))
        print("FP: {} | FN: {}".format(len(fp_axis), len(fn_axis)))

        for axis, freq in zip(BINS, fn_freq):
            print("Axis {} mistaken {} times".format(axis, freq))

        # plot_utils.show_histograms(tp=tp_angles,
        #                            tn=tn_angles,
        #                            fp=fp_angles,
        #                            fn=fn_angles,
        #                            bins_count=4,
        #                            title="For digit {}".format(current_digit))


def _show_histogram(tp, tn, fp, fn):
    plot_utils.show_histograms(tp=_get_digits(tp),
                               tn=_get_digits(tn),
                               fp=_get_digits(fp),
                               fn=_get_digits(fn),
                               bins_count=10,
                               title="All digits")

    plot_utils.show_histograms(tp=_get_angles(tp),
                               tn=_get_angles(tn),
                               fp=_get_angles(fp),
                               fn=_get_angles(fn),
                               bins_count=4,
                               title="All angles")


# todo napisac do tego testy
def _get_frequency(angels, bins=[0, 90, 180, 270]):
    result = np.zeros(len(bins))

    for i, bin in enumerate(bins):
        result[i] = sum([float(angle) == float(bin) for angle in angels])

    return result


def _get_digits(list_of_tuples):
    return [digit for digit, angle in list_of_tuples]


def _get_angles(list_of_tuples):
    return [angle for digit, angle in list_of_tuples]


def _get_confusion_matrix(dataset, predictions):
    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []

    for i, prediction in enumerate(predictions):
        if prediction == 1 and prediction == dataset.get_label(i):
            true_positives.append((dataset.get_digit(i), dataset.get_angle(i)))
        elif prediction == 0 and prediction == dataset.get_label(i):
            true_negatives.append((dataset.get_digit(i), dataset.get_angle(i)))
        elif prediction == 1:
            _save(dataset.get_image(i), dir=FALSE_POSITIVES_DIR,i=i)
            false_positives.append((dataset.get_digit(i), dataset.get_angle(i)))
        else:
            _save(dataset.get_image(i), dir=FALSE_NEGATIVES_DIR,i=i)
            false_negatives.append((dataset.get_digit(i), dataset.get_angle(i)))

    return true_positives, true_negatives, false_positives, false_negatives


def _save(image, dir, i):
    return
    full_path = os.path.join(PREDICTIONS_PATH, dir)
    full_path = os.path.join(full_path, str(i) + '.png')
    plot_utils.show_image(image, full_path)


