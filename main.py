from pathlib import Path
from src.data_generator import generate_synthetic_energy_data
from src.data_loader import load_energy_data
from src.preprocessing import clean_energy_data
from src.features import engineer_features
from src.modeling import train_and_evaluate_model
from src.visualization import save_visualizations


def main():
    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / "data"
    data_dir.mkdir(exist_ok=True)

    csv_path = data_dir / "synthetic_energy_data.csv"

    if not csv_path.exists():
        df = generate_synthetic_energy_data()
        df.to_csv(csv_path, index=False)
        print(f"Generated dataset at: {csv_path}")
    else:
        print(f"Using existing dataset: {csv_path}")

    df = load_energy_data(csv_path)
    df = clean_energy_data(df)
    df = engineer_features(df)

    metrics, model, X_test, y_test, y_pred = train_and_evaluate_model(df)
    save_visualizations(df, y_test, y_pred, model)

    print("\nForecasting system completed successfully.")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"R2 Score: {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
