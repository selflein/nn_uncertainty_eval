import numpy as np
from sklearn.metrics import confusion_matrix


def brier_score(labels, probs):
    probs = probs.copy()
    probs[np.arange(len(probs)), labels] -= 1
    score = (probs ** 2).sum(1).mean(0)
    return score


def brier_decomposition(labels, probs):
    """Compute the decompositon of the Brier score into its three components
    uncertainty (UNC), reliability (REL) and resolution (RES).

    Brier score is given by `BS = REL - RES + UNC`. The decomposition requires partioning
    into discrete events. Partioning into probability classes `M_k` is done for `p_k > p_i`
    for all `i!=k`. This induces a error when compared to the Brier score.

    For more information on the partioning see
    Murphy, A. H. (1973). A New Vector Partition of the Probability Score, Journal of Applied Meteorology and Climatology, 12(4)

    Args:
        labels: Numpy array of shape (num_preds,) containing the groundtruth
         class in range [0, n_classes - 1].
        probs: Numpy array of shape (num_preds, n_classes) containing predicted
         probabilities for the classes.

    Returns:
        (uncertainty, resolution, relability): Additive components of the Brier
         score decomposition.
    """
    preds = np.argmax(probs, axis=1)
    conf_mat = confusion_matrix(labels, preds, labels=np.arange(probs.shape[1]))

    pbar = np.sum(conf_mat, axis=0)
    pbar = pbar / pbar.sum()

    dist_weights = np.sum(conf_mat, axis=1)
    dist_weights = dist_weights / dist_weights.sum()

    dist_mean = conf_mat / (np.sum(conf_mat, axis=1)[:, None] + 1e-7)

    uncertainty = np.sum(pbar * (1 - pbar))

    resolution = (pbar[:, None] - dist_mean) ** 2
    resolution = np.sum(dist_weights * np.sum(resolution, axis=1))

    prob_true = np.take(dist_mean, preds, axis=0)
    reliability = np.sum((prob_true - probs) ** 2, axis=1)
    reliability = np.mean(reliability)

    return uncertainty, resolution, reliability

