'''
PART 1: ETL the two datasets and save each in `data/` as .csv's
'''

import pandas as pd
from pathlib import Path

DATA_DIR = Path('data')

def run_etl() -> dict[str, Path]:
    """
    Execute ETL for raw inputs and persist them to `data/`.

    Returns
    -------
    dict[str, Path]
        Mapping of dataset names to their saved CSV paths:
        - 'pred_universe_raw'
        - 'arrest_events_raw'
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pred_universe_raw = pd.read_csv(
        "https://www.dropbox.com/scl/fi/69syqjo6pfrt9123rubio/universe_lab6.feather?rlkey=h2gt4o6z9r5649wo6h6ud6dce&dl=1"
    )
    arrest_events_raw = pd.read_csv(
        "https://www.dropbox.com/scl/fi/wv9kthwbj4ahzli3edrd7/arrest_events_lab6.feather?rlkey=mhxozpazqjgmo6qqahc2vd0xp&dl=1"
    )
    pred_universe_raw["arrest_date_univ"] = pd.to_datetime(pred_universe_raw.filing_date)
    arrest_events_raw["arrest_date_event"] = pd.to_datetime(arrest_events_raw.filing_date)
    pred_universe_raw.drop(columns=["filing_date"], inplace=True)
    arrest_events_raw.drop(columns=["filing_date"], inplace=True)

    # set paths and directories for data files
    pred_path = DATA_DIR / "pred_universe_raw.csv"
    events_path = DATA_DIR / "arrest_events_raw.csv"
    # convert to csv
    pred_universe_raw.to_csv(pred_path, index=False)
    arrest_events_raw.to_csv(events_path, index=False)

    print(f"Saved: {pred_path}")
    print(f"Saved: {events_path}")

    return {"pred_universe_raw": pred_path, "arrest_events_raw": events_path}


if __name__ == "__main__":
    raise SystemExit("Import and call run_etl() from main.py")