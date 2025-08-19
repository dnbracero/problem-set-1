'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC
from pathlib import Path

# CONSTANTS
# ---- Configuration ----
RANDOM_STATE = 414
features = ["num_fel_arrests_last_year", "current_charge_felony"]
TARGET = "y"

DATA_DIR = Path("data")
TRAIN_IN_DEFAULT = DATA_DIR / "df_arrests_train_lr.csv"  # from PART 3
TEST_IN_DEFAULT = DATA_DIR / "df_arrests_test_lr.csv"    # from PART 3

TRAIN_OUT = DATA_DIR / "df_arrests_train_dt.csv"
TEST_OUT = DATA_DIR / "df_arrests_test_dt.csv"


def validate_inputs(df):
    """Ensure required columns exist in the provided DataFrame."""
    required = set(features + [TARGET])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {sorted(missing)}")


def prepare_features(df):
    """Select and coerce feature columns to numeric; fill NAs with 0."""
    X = df[features].copy()
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.fillna(0)


def prepare_target(df):
    """Coerce target to numeric (0/1)."""
    return pd.to_numeric(df[TARGET], errors="coerce")


def build_dt_model():
    """Create a base Decision Tree classifier."""
    return DTC(
        criterion="gini",
        random_state=RANDOM_STATE,
    )


def regularization_interpretation_for_depth(best_depth, grid):
    """
    Interpret tree depth in terms of regularization.
    Smaller depth -> stronger regularization; larger depth -> weaker regularization.
    """
    smallest, largest = min(grid), max(grid)
    if best_depth == smallest:
        return "most regularization"
    if best_depth == largest:
        return "least regularization"
    return "in the middle"


def load_lr_splits(df_arrests_train=None, df_arrests_test=None):
    """
    If train/test are provided, use them. Otherwise, load the LR splits saved in PART 3.
    """
    if df_arrests_train is not None and df_arrests_test is not None:
        return df_arrests_train.copy(), df_arrests_test.copy()

    if not TRAIN_IN_DEFAULT.exists() or not TEST_IN_DEFAULT.exists():
        raise FileNotFoundError(
            "Train/test DataFrames not provided and LR split files not found at "
            f"{TRAIN_IN_DEFAULT} and {TEST_IN_DEFAULT}. Run PART 3 first or pass DataFrames."
        )

    train = pd.read_csv(TRAIN_IN_DEFAULT)
    test = pd.read_csv(TEST_IN_DEFAULT)
    return train, test


def run_decision_tree(df_arrests_train=None, df_arrests_test=None):
    """
    Train a Decision Tree (with max_depth grid search), predict on the test set, and return results.

    Parameters
    ----------
    df_arrests_train : pd.DataFrame, optional
        Training split with at least the target and feature columns. If None, reads saved LR split.
    df_arrests_test : pd.DataFrame, optional
        Test split with at least the target and feature columns. If None, reads saved LR split.

    Returns
    -------
    df_arrests_train : pd.DataFrame
        Unmodified training set (for completeness/consistency with other parts).
    df_arrests_test_with_pred : pd.DataFrame
        Test set augmented with 'pred_dt' (P(y=1)).
    gs_cv_dt : GridSearchCV
        Fitted GridSearchCV object for the Decision Tree.
    """
    # Resolve inputs
    df_train, df_test = load_lr_splits(df_arrests_train, df_arrests_test)

    # Validate
    validate_inputs(df_train)
    validate_inputs(df_test)

    # Features/target
    X_train = prepare_features(df_train)
    y_train = prepare_target(df_train)
    X_test = prepare_features(df_test)

    # Parameter grid: three depths
    param_grid_dt = {"max_depth": [2, 4, 6]}

    # Model + CV
    dt_model = build_dt_model()
    cv = KFold_strat(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs_cv_dt = GridSearchCV(
        estimator=dt_model,
        param_grid=param_grid_dt,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    gs_cv_dt.fit(X_train, y_train)

    # Required prints
    best_depth = int(gs_cv_dt.best_params_["max_depth"])
    interpretation = regularization_interpretation_for_depth(best_depth, param_grid_dt["max_depth"])
    print("What was the optimal value for max_depth?")
    print(f"Answer: {best_depth}")
    print("Did it have the most or least regularization? Or in the middle?")
    print(f"Answer: {interpretation}")

    # Predict probabilities for the test set (positive class)
    df_test = df_test.copy()
    if hasattr(gs_cv_dt.best_estimator_, "predict_proba"):
        df_test["pred_dt"] = gs_cv_dt.predict_proba(X_test)[:, 1]
    else:
        # Fallback (rare for DTC): use decision_function then map to [0,1] via min-max
        raw = gs_cv_dt.decision_function(X_test).astype(float)
        df_test["pred_dt"] = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)

    # Save outputs for PART 5
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(TRAIN_OUT, index=False)
    df_test.to_csv(TEST_OUT, index=False)
    print(f"Decision Tree splits saved:\n - {TRAIN_OUT}\n - {TEST_OUT}")

    return df_train, df_test, gs_cv_dt


if __name__ == "__main__":
    raise SystemExit("Call run_decision_tree(train_df, test_df) from main.py (after PART 3).")