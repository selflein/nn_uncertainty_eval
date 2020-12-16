import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, average_precision_score

from uncertainty_eval.metrics.calibration_error import calc_bins, classification_calibration


def draw_reliability_graph(labels, probs, num_bins, ax=None):
    ece, mce = classification_calibration(labels, probs, num_bins)
    bins, _, bin_accs, _, _ = calc_bins(labels, probs, num_bins)

    if ax is None:
        _, ax = plt.subplots()

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")

    ax.grid(color="gray", linestyle="dashed")

    ax.bar(bins, bins, width=0.1, alpha=0.3, edgecolor="black", color="r", hatch="\\")

    ax.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor="black", color="b")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=2)
    ax.set_aspect("equal", adjustable="box")

    ECE_patch = mpatches.Patch(color="green", label="ECE = {:.2f}%".format(ece * 100))
    MCE_patch = mpatches.Patch(color="red", label="MCE = {:.2f}%".format(mce * 100))
    ax.legend(handles=[ECE_patch, MCE_patch])

    return ax


def plot_classification_curve(measure: str, y_true, y_score, pos_label=None, ax=None):
    if measure == "roc":
        x, y, _ = roc_curve(y_true, y_score, pos_label=pos_label)
        x_label = "FPR"
        y_label = "TPR"
        value = roc_auc_score(y_true, y_score)
    elif measure == "pr":
        y, x, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label)
        x_label = "Recall"
        y_label = "Precision"
        value = average_precision_score(y_true, y_score)
    else:
        raise ValueError(f"Measure {measure} not implemented.")
    
    if ax is None:
        _, ax = plt.subplots()
    
    ax.plot(x, y)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"AU{measure.upper()}: {value * 100:.02f}")
    return ax
