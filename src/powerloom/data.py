from pathlib import Path
import pandas as pd


def remove_duplicates(data):
    """Remove exact duplicate rows."""
    return data.drop_duplicates().copy()


def remove_leaking_column(data, leaking_col):
    """Remove the leakage column if it exists."""
    if leaking_col in data.columns:
        data = data.drop(columns=[leaking_col])
    return data


def replace_sentinel(data, column, value):
    """Replace sentinel values with missing values."""
    count = int((data[column] == value).sum())
    data.loc[data[column] == value, column] = pd.NA
    return data, count


def correct_stuck_sensor(data, column, value, start_date, end_date):
    """Replace stuck sensor readings within the specified date range."""
    mask = (
        (data["date"] >= start_date)
        & (data["date"] <= end_date)
        & (data[column] == value)
    )
    count = int(mask.sum())
    data.loc[mask, column] = pd.NA
    return data, count


def correct_target_typos(data, target_col, threshold):
    """Correct unusually large target values by dividing them by ten."""
    mask = data[target_col] > threshold
    count = int(mask.sum())
    data.loc[mask, target_col] = data.loc[mask, target_col] / 10
    return data, count


def normalize_categories(data, columns):
    """Trim whitespace and lowercase categorical values."""
    for column in columns:
        if column in data.columns:
            data[column] = data[column].astype("string").str.strip().str.lower()
    return data


def impute_median(data, column):
    """Fill missing values in a numeric column with its median."""
    data[column] = data[column].fillna(data[column].median())
    return data


def save_clean_data(data, output_path):
    """Save cleaned data, creating the parent directory if needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)