'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import etl
import preprocessing
# import logistic_regression
# import decision_tree
# import calibration_plot


# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    data_paths = etl.run_etl()
    print('ETL complete. Files saved:')
    for name, path in data_paths.items():
        print(f' - {name}: {path}')

    # PART 2: Call functions/instanciate objects from preprocessing
    df_arrests = preprocessing.run_preprocessing(data_paths)
    print('Preprocessing complete.')

    # PART 3: Call functions/instanciate objects from logistic_regression

    # PART 4: Call functions/instanciate objects from decision_tree

    # PART 5: Call functions/instanciate objects from calibration_plot


if __name__ == "__main__":
    main()