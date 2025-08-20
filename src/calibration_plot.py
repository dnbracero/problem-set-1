'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

PLOT_DIR = Path('plot')

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).
    y_prob : array-like
        Predicted probabilities for the positive class.
    n_bins : int, default=10
        Number of bins to divide the data for calibration.

    Returns
    -------
    None
    """
    # Calculate calibration values
    # Note: sklearn returns (prob_true, prob_pred).
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)

    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.plot(prob_true, bin_means, marker="o", label="Model")

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")

    # Save plot pngs in plot/
    out_dir = PLOT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calibration_plot.png"
    i = 2
    while out_path.exists():
        out_path = out_dir / f"calibration_plot-{i}.png"
        i += 1
    plt.gcf().savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved calibration plot to: {out_path}")

    plt.show()


def expected_calibration_error(y_true, y_prob, n_bins=5):
    """
    Expected Calibration Error (uniform-width bins). Lower is better (more calibrated).

    Parameters
    ----------
    y_true : array-like
    y_prob : array-like
    n_bins : int, default=5

    Returns
    -------
    float
        ECE score. NaN if y_true is empty.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_true.size == 0:
        return float("nan")

    # Uniform-width bins from 0 to 1
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Digitize with right-inclusive bins to align with many plotting conventions
    idx = np.digitize(y_prob, edges, right=True) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    N = y_true.shape[0]
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        n_b = int(np.sum(mask))
        if n_b == 0:
            continue
        conf_b = float(np.mean(y_prob[mask]))
        acc_b = float(np.mean(y_true[mask]))
        ece += abs(acc_b - conf_b) * (n_b / N)

    return float(ece)


if __name__ == "__main__":
    raise SystemExit("Import and call from main.py")