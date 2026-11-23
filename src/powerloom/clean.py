"""Data cleaning pipeline for MP243 Power Loom Output."""

from pathlib import Path
import pandas as pd
import yaml

from powerloom.data import (
    remove_duplicates,
    remove_leaking_column,
    replace_sentinel,
    correct_stuck_sensor,
    correct_target_typos,
    normalize_categories,
    impute_median,
    save_clean_data,
)

ROOT = Path(__file__).resolve().parents[2]


def load_params():
    with open(ROOT / "params.yaml", "r") as file:
        return yaml.safe_load(file)


def clean_train():
    params = load_params()

    train_path = ROOT / params["paths"]["train_raw"]
    output_path = ROOT / params["paths"]["train_clean"]

    data = pd.read_csv(train_path)
    original_rows = len(data)

    data = remove_duplicates(data)
    rows_after_duplicates = len(data)

    data = remove_leaking_column(data, params["data"]["leaking_col"])

    data, sentinel_count = replace_sentinel(
        data, "thread_count", 99
    )

    data, stuck_count = correct_stuck_sensor(
        data,
        params["data"]["stuck_sensor_col"],
        params["data"]["stuck_sensor_val"],
        params["data"]["stuck_start"],
        params["data"]["stuck_end"],
    )

    data, typo_count = correct_target_typos(
        data,
        params["data"]["target_col"],
        params["data"]["typo_threshold"],
    )

    data = normalize_categories(data, ["fabric", "shed"])

    data = impute_median(data, "thread_count")

    save_clean_data(data, output_path)

    print("Cleaning completed!")
    print(f"Original rows: {original_rows}")
    print(f"Rows after duplicate removal: {rows_after_duplicates}")
    print(f"Duplicates removed: {original_rows - rows_after_duplicates}")
    print(f"Target typos corrected: {typo_count}")
    print(f"Sentinel values replaced: {sentinel_count}")
    print(f"Stuck sensor readings corrected: {stuck_count}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    clean_train()