'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages

import pandas as pd
from pathlib import Path 

# Your code here

def load_data(data_paths: dict) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """
    Resolve file paths from ETL output and load the CSVs.

    Parameters
    ----------
    data_paths : dict
        Expected keys:
          - 'pred_universe_raw'
          - 'arrest_events_raw'
        Optional:
          - 'df_arrests' (output path override)

    Returns
    -------
    (pred, events, out_path) : tuple[pd.DataFrame, pd.DataFrame, Path]
    """
    def get_path(d: dict, keys: list[str]) -> str:
        for k in keys:
            if k in d and d[k]:
                return d[k]
        raise KeyError(f"Expected one of {keys} in data_paths. Got keys: {list(d.keys())}")

    pred_path = get_path(data_paths, ["pred_universe_raw", "pred_universe", "pred"])
    events_path = get_path(data_paths, ["arrest_events_raw", "arrest_events", "arrests"])

    out_path = Path(data_paths.get("df_arrests", "data/df_arrests.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(pred_path)
    events = pd.read_csv(events_path)
    return pred, events, out_path


def normalize_data(pred: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure required columns exist and standardize date fields.

    Parameters
    ----------
    pred : pd.DataFrame
    events : pd.DataFrame

    Returns
    -------
    (pred_norm, events_norm) : tuple[pd.DataFrame, pd.DataFrame]
    """
    pred = pred.copy()
    events = events.copy()

    # Dates
    if "arrest_date_univ" not in pred.columns:
        if "arrest_date" in pred.columns:
            pred["arrest_date_univ"] = pred["arrest_date"]
        else:
            raise KeyError("pred_universe is missing 'arrest_date_univ' (or fallback 'arrest_date').")
    pred["arrest_date_univ"] = pd.to_datetime(pred["arrest_date_univ"])

    if "arrest_date_event" not in events.columns:
        if "arrest_date" in events.columns:
            events["arrest_date_event"] = events["arrest_date"]
        else:
            raise KeyError("arrest_events is missing 'arrest_date_event' (or fallback 'arrest_date').")
    events["arrest_date_event"] = pd.to_datetime(events["arrest_date_event"])

    # Keys
    if "person_id" not in pred.columns or "person_id" not in events.columns:
        raise KeyError("Both pred_universe and arrest_events must contain 'person_id'.")

    # Stable per-row id for window grouping
    if "row_id" not in pred.columns:
        pred = pred.reset_index().rename(columns={"index": "row_id"})

    return pred, events


def add_is_felony_flag(events: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean 'is_felony' based on common severity columns.

    Parameters
    ----------
    events : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    events = events.copy()
    felony_col = None
    for c in ["charge_degree", "offense_level", "charge_severity", "charge_class"]:
        if c in events.columns:
            felony_col = c
            break

    if felony_col is not None:
        norm = events[felony_col].astype(str).str.strip().str.lower()
        events["is_felony"] = norm.isin(["felony", "f"])
    else:
        events["is_felony"] = False

    return events


def create_features(pred: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    Create required features on the prediction universe.

    Returns
    -------
    pd.DataFrame
        Copy of pred with: current_charge_felony, num_fel_arrests_last_year, y
    """
    pr = pred.copy()
    ev = events.copy()

    # (A) current_charge_felony
    if "arrest_id" in pr.columns and "arrest_id" in ev.columns:
        current_map = ev.set_index("arrest_id")["is_felony"]
        pr["current_charge_felony"] = pr["arrest_id"].map(current_map).fillna(False).astype(int)
    else:
        # Fallback: same-day heuristic
        ev["_event_day"] = ev["arrest_date_event"].dt.date
        pr["_current_day"] = pr["arrest_date_univ"].dt.date
        same_day_felony = (
            ev.groupby(["person_id", "_event_day"])["is_felony"].any().rename("same_day_felony")
        )
        idx = pd.MultiIndex.from_frame(pr[["person_id", "_current_day"]])
        pr["current_charge_felony"] = (
            pd.Series(same_day_felony.reindex(idx).values, index=pr.index).fillna(False).astype(int)
        )

    # Build pairs for windowed counts
    pairs = pr[["row_id", "person_id", "arrest_date_univ"]].merge(
        ev[["person_id", "arrest_date_event", "is_felony"]],
        on="person_id",
        how="left",
    )

    # (B) num_fel_arrests_last_year: [current -365, current -1]
    past_mask = (
        pairs["is_felony"].fillna(False)
        & (pairs["arrest_date_event"] >= pairs["arrest_date_univ"] - pd.Timedelta(days=365))
        & (pairs["arrest_date_event"] <= pairs["arrest_date_univ"] - pd.Timedelta(days=1))
    )
    past_counts = past_mask.groupby(pairs["row_id"]).sum().astype(int)
    pr["num_fel_arrests_last_year"] = pr["row_id"].map(past_counts).fillna(0).astype(int)

    # (C) y label (future window): [current +1, current +365]
    future_mask = (
        pairs["is_felony"].fillna(False)
        & (pairs["arrest_date_event"] >= pairs["arrest_date_univ"] + pd.Timedelta(days=1))
        & (pairs["arrest_date_event"] <= pairs["arrest_date_univ"] + pd.Timedelta(days=365))
    )
    future_any = future_mask.groupby(pairs["row_id"]).any()
    pr["y"] = pr["row_id"].map(future_any).fillna(False).astype(int)

    return pr


def report_metrics(pred_with_feats: pd.DataFrame) -> None:
    """
    Print required summary metrics and a sample head for inspection.
    """
    print("What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?")
    print(f"Answer: {pred_with_feats['y'].mean():.3%}")

    print("What share of current charges are felonies?")
    print(f"Answer: {pred_with_feats['current_charge_felony'].mean():.3%}")

    print("What is the average number of felony arrests in the last year?")
    print(f"Answer: {pred_with_feats['num_fel_arrests_last_year'].mean():.3f}")

    print("Mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()")
    print(f"{pred_with_feats['num_fel_arrests_last_year'].mean():.3f}")

    print("\npred_universe.head():")
    print(pred_with_feats.head())


def create_df_arrests(pred_with_feats: pd.DataFrame, events: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """
    Full outer join on person_id, save, and return df_arrests.

    Returns
    -------
    pd.DataFrame
    """
    df_arrests = pred_with_feats.merge(
        events,
        on="person_id",
        how="outer",
        suffixes=("_univ", "_event"),
    )
    df_arrests.to_csv(out_path, index=False)
    return df_arrests


def run_preprocessing(data_paths: dict) -> pd.DataFrame:
    """
    Orchestrate preprocessing: load, normalize, feature-create, report, and persist.

    Returns
    -------
    pd.DataFrame
        The `df_arrests` table saved to disk.
    """
    pred, events, out_path = load_data(data_paths)
    pred, events = normalize_data(pred, events)
    events = add_is_felony_flag(events)
    pred = create_features(pred, events)
    report_metrics(pred)
    df_arrests = create_df_arrests(pred, events, out_path)
    return df_arrests


if __name__ == "__main__":
    raise SystemExit("Call run_preprocessing(data_paths) from main.py")