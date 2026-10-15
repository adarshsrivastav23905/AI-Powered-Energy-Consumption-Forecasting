from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_visualizations(df, y_test, y_pred, model):
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values[:200], label="Actual", color="blue")
    plt.plot(y_pred[:200], label="Predicted", color="red", linestyle="--")
    plt.title("Actual vs Predicted Energy Consumption")
    plt.xlabel("Samples")
    plt.ylabel("Energy (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png")
    plt.close()

    feature_names = model.feature_names_in_
    feature_importance = model.feature_importances_

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importance)
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png")
    plt.close()

    print(f"Saved visualizations to: {output_dir}")
