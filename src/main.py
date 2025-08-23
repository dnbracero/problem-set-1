'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import numpy as np
import etl
import preprocessing
import logistic_regression
import decision_tree
import calibration_plot


# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    data_paths = etl.run_etl()
    print("ETL complete. Files saved:")
    for name, path in data_paths.items():
        print(f" - {name}: {path}")

    # PART 2: Call functions/instanciate objects from preprocessing
    df_arrests = preprocessing.run_preprocessing(data_paths)
    print("Pre-processing complete.")

    # PART 3: Call functions/instanciate objects from logistic_regression
    df_arrests_train, df_arrests_test, gs_cv = logistic_regression.run_logistic_regression(df_arrests)
    print("Logistic Regression complete.")
    print(f"Train shape: {df_arrests_train.shape} | Test shape: {df_arrests_test.shape}")
    print(f"Best params from GridSearchCV: {gs_cv.best_params_}")

    # PART 4: Call functions/instanciate objects from decision_tree
    df_arrests_train_dt, df_arrests_test_dt, gs_cv_dt = decision_tree.run_decision_tree(
        df_arrests_train, df_arrests_test
    )
    print("Decision Tree complete.")
    print(f"Best params from GridSearchCV (DT): {gs_cv_dt.best_params_}")

    # PART 5: Call functions/instanciate objects from calibration_plot
    # Create calibration curves for LR and DT with n_bins=5
    calibration_plot.calibration_plot(
        y_true=df_arrests_test["y"].values,
        y_prob=df_arrests_test["pred_lr"].values,
        n_bins=5,
    )
    calibration_plot.calibration_plot(
        y_true=df_arrests_test_dt["y"].values,
        y_prob=df_arrests_test_dt["pred_dt"].values,
        n_bins=5,
    )

    print("Which model is more calibrated?")
    ece_lr = calibration_plot.expected_calibration_error(
        df_arrests_test["y"].values, df_arrests_test["pred_lr"].values, n_bins=5
    )
    ece_dt = calibration_plot.expected_calibration_error(
        df_arrests_test_dt["y"].values, df_arrests_test_dt["pred_dt"].values, n_bins=5
    )
    if np.isfinite(ece_lr) and np.isfinite(ece_dt):
        if ece_lr < ece_dt:
            print(f"Answer: Logistic Regression (ECE — LR: {ece_lr:.4f}, DT: {ece_dt:.4f})")
        elif ece_dt < ece_lr:
            print(f"Answer: Decision Tree (ECE — LR: {ece_lr:.4f}, DT: {ece_dt:.4f})")
        else:
            print(f"Answer: Tie (ECE — LR: {ece_lr:.4f}, DT: {ece_dt:.4f})")
    else:
        print("Answer: Unable to determine (ECE undefined).")


if __name__ == "__main__":
    main()