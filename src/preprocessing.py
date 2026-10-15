import pandas as pd


def clean_energy_data(df):
    df = df.copy()
    df = df.drop_duplicates().sort_values("timestamp").reset_index(drop=True)

    numeric_columns = ["energy_kwh", "temperature_c", "occupancy_score"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    return df
