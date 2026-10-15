from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_generator import generate_synthetic_energy_data

IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)


def build_assets():
    df = generate_synthetic_energy_data()
    sample = df.head(24)

    plt.figure(figsize=(12, 4))
    plt.plot(sample["timestamp"], sample["energy_kwh"], marker="o")
    plt.title("Dataset Preview")
    plt.xlabel("Timestamp")
    plt.ylabel("Energy (kWh)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "dataset_preview.png", dpi=200)
    plt.close()

    hourly = df.set_index("timestamp").resample("D").mean()
    plt.figure(figsize=(12, 5))
    plt.plot(hourly.index, hourly["energy_kwh"], color="tab:green")
    plt.title("Daily Energy Demand Trend")
    plt.xlabel("Date")
    plt.ylabel("Average Energy (kWh)")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "daily_energy_trend.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.hist(df["energy_kwh"], bins=30, color="tab:purple", edgecolor="black")
    plt.title("Energy Consumption Distribution")
    plt.xlabel("Energy (kWh)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "energy_distribution.png", dpi=200)
    plt.close()

    print(f"Saved GitHub-ready asset files in {IMAGES_DIR}")


if __name__ == "__main__":
    build_assets()
