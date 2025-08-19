'''
PART 1: ETL the two datasets and save each in `data/` as .csv's
'''

import pandas as pd
from pathlib import Path

def run_etl():
    """
    Loads datasets, creates date fields, drops filing_date,
    and saves CSVs to data/ as pred_universe_raw.csv and arrest_events_raw.csv.
    """
    pred_universe_raw = pd.read_csv('https://www.dropbox.com/scl/fi/69syqjo6pfrt9123rubio/universe_lab6.feather?rlkey=h2gt4o6z9r5649wo6h6ud6dce&dl=1')
    arrest_events_raw = pd.read_csv('https://www.dropbox.com/scl/fi/wv9kthwbj4ahzli3edrd7/arrest_events_lab6.feather?rlkey=mhxozpazqjgmo6qqahc2vd0xp&dl=1')
    pred_universe_raw['arrest_date_univ'] = pd.to_datetime(pred_universe_raw.filing_date)
    arrest_events_raw['arrest_date_event'] = pd.to_datetime(arrest_events_raw.filing_date)
    pred_universe_raw.drop(columns=['filing_date'], inplace=True)
    arrest_events_raw.drop(columns=['filing_date'], inplace=True)

    # Save both data frames to `data/` -> 'pred_universe_raw.csv', 'arrest_events_raw.csv'
    data_dir = Path('data')
    data_dir.mkdir(parents=True, exist_ok=True)
    pred_path = data_dir / 'pred_universe_raw.csv'
    arrest_path = data_dir / 'arrest_events_raw.csv'

    pred_universe_raw.to_csv(pred_path, index=False)
    arrest_events_raw.to_csv(arrest_path, index=False)

    print(f'Saved: {pred_path}')
    print(f'Saved: {arrest_path}')

    return {'pred_universe_raw': pred_path, 'arrest_events_raw': arrest_path}

if __name__ == "__main__":
    run_etl()