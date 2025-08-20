'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: num_fel_arrests_last_year, current_charge_felony
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression as lr
from pathlib import Path


# Your code here

# constants
RANDOM_STATE: int = 414
TEST_SIZE: float = 0.30

# feature names
features = ["num_fel_arrests_last_year", "current_charge_felony"]
TARGET = "y"

DATA_DIR = Path("data")
TRAIN_OUT = DATA_DIR / "df_arrests_train_lr.csv"
TEST_OUT = DATA_DIR / "df_arrests_test_lr.csv"


def validate_inputs(df_arrests: pd.DataFrame) -> None:
    """
    Ensure that required columns are present.

    Parameters
    ----------
    df_arrests : pd.DataFrame
    """
    required = set(features + [TARGET])
    missing = required.difference(df_arrests.columns)
    if missing:
        raise ValueError(f"df_arrests is missing required columns: {sorted(missing)}")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and clean feature columns: numeric coercion and NA fill.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    X = df[features].copy()
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.fillna(0)


def prepare_target(df: pd.DataFrame) -> pd.Series:
    """
    Prepare binary target (coerce to numeric).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.Series
    """
    y = pd.to_numeric(df[TARGET], errors="coerce")
    return y


def split_data(df_arrests: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test split (by y) and return full-row DataFrames.
    Drops rows where y is NA before splitting.

    Parameters
    ----------
    df_arrests : pd.DataFrame

    Returns
    -------
    (train_df, test_df) : tuple[pd.DataFrame, pd.DataFrame]
    """
    df = df_arrests[df_arrests[TARGET].notna()].copy()
    y = prepare_target(df)

    idx_train, idx_test = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        shuffle=True,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return df.loc[idx_train].copy(), df.loc[idx_test].copy()


def build_lr_model() -> lr:
    """
    Create the base Logistic Regression model.

    Returns
    -------
    sklearn.linear_model.LogisticRegression
    """
    return lr(
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )


def regularization_interpretation(best_c: float, grid: list[float]) -> str:
    """
    Interpret C: smaller C -> stronger regularization.

    Parameters
    ----------
    best_c : float
    grid : list[float]

    Returns
    -------
    str
        'most regularization', 'least regularization', or 'in the middle'
    """
    smallest, largest = min(grid), max(grid)
    if best_c == smallest:
        return "most regularization"
    if best_c == largest:
        return "least regularization"
    return "in the middle"


def run_logistic_regression(
    df_arrests: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, GridSearchCV]:
    """
    Train a logistic regression model with grid search on C, predict on test set,
    print answers, and return/save the train/test DataFrames.

    Parameters
    ----------
    df_arrests : pd.DataFrame
        DataFrame produced in PART 2 with features and label.

    Returns
    -------
    (df_arrests_train, df_arrests_test_with_pred, gs_cv) : tuple[pd.DataFrame, pd.DataFrame, GridSearchCV]
    """
    # Validate inputs
    validate_inputs(df_arrests)

    # Split
    df_arrests_train, df_arrests_test = split_data(df_arrests)
    X_train = prepare_features(df_arrests_train)
    y_train = prepare_target(df_arrests_train)

    # Grid + model
    param_grid = {"C": [0.01, 0.1, 1.0]}
    lr_model = build_lr_model()

    cv = KFold_strat(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs_cv = GridSearchCV(
        estimator=lr_model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    gs_cv.fit(X_train, y_train)

    # Required prints
    best_c = float(gs_cv.best_params_["C"])
    interpretation = regularization_interpretation(best_c, param_grid["C"])
    print("What was the optimal value for C?")
    print(f"Answer: {best_c}")
    print("Did it have the most or least regularization? Or in the middle?")
    print(f"Answer: {interpretation}")

    # Predict for the test set (P(y=1))
    X_test = prepare_features(df_arrests_test)
    df_arrests_test = df_arrests_test.copy()
    df_arrests_test["pred_lr"] = gs_cv.predict_proba(X_test)[:, 1]

    # Persist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_arrests_train.to_csv(TRAIN_OUT, index=False)
    df_arrests_test.to_csv(TEST_OUT, index=False)
    print(f"Train/Test splits saved:\n - {TRAIN_OUT}\n - {TEST_OUT}")

    return df_arrests_train, df_arrests_test, gs_cv


if __name__ == "__main__":
    raise SystemExit("Call run_logistic_regression(df_arrests) from main.py")