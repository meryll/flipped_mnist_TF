import matplotlib.pyplot as plt
import numpy as np


def show_image(image, path=None):
    plt.imshow(np.reshape(image, [28, 28]), cmap='gray')
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close()


def show_learniing_curve(acc_train, acc_test):
    x = np.arange(len(acc_train))

    plt.plot(x, acc_train)
    plt.plot(x, acc_test)

    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def show_histogram(values, title):
    plt.hist(values, normed=True, bins=10)
    plt.ylabel('');


def show_histograms(tp, tn, fp, fn, bins_count, title):
    bins = np.arange(bins_count) - 0.5
    width = 0.5

    f, axarr = plt.subplots(2, 2)
    plt.title(title)

    axarr[0, 0].hist(tp, normed=False, bins=bins, rwidth=width)
    axarr[0, 0].set_title('True Positives')
    axarr[0, 1].hist(tn, normed=False, bins=bins, rwidth=width)
    axarr[0, 1].set_title('True Negatives')
    axarr[1, 0].hist(fp, normed=False, bins=bins, rwidth=width)
    axarr[1, 0].set_title('False Positives')
    axarr[1, 1].hist(fn, normed=False, bins=bins, rwidth=width)
    axarr[1, 1].set_title('False Negatives')

    plt.show()
