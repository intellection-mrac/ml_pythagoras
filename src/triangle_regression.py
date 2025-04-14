import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# Force a non-interactive backend to avoid threading issues
matplotlib.use('Agg')

# Constants
N_SAMPLES = 1000
LEG_RANGE = (1, 100)
TEST_SIZE = 0.2
RANDOM_SEED = 42
DATA_PATH = Path("data/triangle_dataset.csv")
PLOT_PATH = Path("plots/actual_vs_predicted.png")

def generate_triangle_data(n_samples: int, leg_range: tuple) -> pd.DataFrame:
    """Generate dataset of right-angle triangles."""
    np.random.seed(RANDOM_SEED)
    a = np.random.uniform(*leg_range, n_samples)
    b = np.random.uniform(*leg_range, n_samples)
    c = np.sqrt(a**2 + b**2)
    return pd.DataFrame({'a': a, 'b': b, 'c': c})

def save_data(data: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)

def train_and_evaluate(data: pd.DataFrame) -> dict:
    """Train model and return evaluation metrics."""
    X = data[['a', 'b']].to_numpy()
    y = data['c'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2_score(y_test, y_pred),
        'y_test': y_test,
        'y_pred': y_pred
    }

def plot_results(y_test: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    """Plot actual vs. predicted hypotenuse."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Hypotenuse')
    plt.ylabel('Predicted Hypotenuse')
    plt.title('Actual vs. Predicted Hypotenuse')
    plt.legend()
    plt.grid(True)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300)
    plt.close()

def main() -> None:
    """Orchestrate the pipeline."""
    # Generate and save dataset
    data = generate_triangle_data(N_SAMPLES, LEG_RANGE)
    save_data(data, DATA_PATH)
    
    # Train and evaluate
    results = train_and_evaluate(data)
    print(f"Evaluation Metrics:")
    print(f"MSE: {results['mse']:.2f}")
    print(f"RMSE: {results['rmse']:.2f}")
    print(f"R²: {results['r2']:.2f}")
    
    # Plot results
    plot_results(results['y_test'], results['y_pred'], PLOT_PATH)

if __name__ == "__main__":
    main()
