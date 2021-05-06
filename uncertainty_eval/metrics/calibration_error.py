import numpy as np


def calc_bins(labels, probs, num_bins=10):
    bins = np.linspace(0.1, 1, num_bins)
    confs = np.max(probs, axis=1)
    binned = np.digitize(confs, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)
    for bin_idx in range(num_bins):
        in_bin = binned == bin_idx
        bin_probs = probs[in_bin]
        if len(bin_probs) > 0:
            bin_sizes[bin_idx] = len(bin_probs)
            bin_confs[bin_idx] = np.mean(confs[in_bin])
            bin_accs[bin_idx] = (np.argmax(bin_probs, axis=1) == labels[in_bin]).mean()
    return bins, binned, bin_accs, bin_confs, bin_sizes


def classification_calibration(labels, probs, num_bins=10):
    _, _, bin_accs, bin_confs, bin_sizes = calc_bins(labels, probs, num_bins)

    cal_errors = np.abs(bin_accs - bin_confs)
    mce = np.max(cal_errors)
    ece = np.sum(cal_errors * (bin_sizes / (np.sum(bin_sizes) + 1e-7)))

    return ece, mce
