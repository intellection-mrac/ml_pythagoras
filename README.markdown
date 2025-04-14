# ML for Robotic Fabrication: Predicting Hypotenuse with Machine Learning

This repository contains the solution for the individual assignment in "ML for Robotic Fabrication," applying machine learning to predict the hypotenuse of a right-angle triangle using Python and scikit-learn.

## Dataset Description

- **File**: `data/triangle_dataset.csv`
- **Details**:
  - Contains 1000 samples of right-angle triangles.
  - Columns:
    - `a`: Length of one leg, randomly sampled from a uniform distribution between 1 and 100.
    - `b`: Length of the other leg, randomly sampled from a uniform distribution between 1 and 100.
    - `c`: Hypotenuse, computed using Pythagoras' theorem (`c = √(a² + b²)`).
  - Generated using `numpy.random.uniform` for randomness and reproducibility (seed=42).
  - Saved as a CSV file for easy access.

## Model Details

- **Model**: Linear Regression
- **Library**: `scikit-learn` (`sklearn.linear_model.LinearRegression`)
- **Features**: Leg lengths `a` and `b`
- **Target**: Hypotenuse `c`
- **Training Process**:
  - Dataset split into 80% training and 20% testing sets using `sklearn.model_selection.train_test_split`.
  - Model trained with default parameters (ordinary least squares).
  - Random seed set to 42 for reproducibility.
- **Implementation**: Code is in `src/triangle_regression.py`, which includes data generation, model training, evaluation, and visualization.

## Evaluation Results

The model was evaluated on the test set (20% of data) with the following metrics:

| Metric | Value |
| --- | --- |
| Mean Squared Error (MSE) | 36.85 |
| Root Mean Squared Error (RMSE) | 6.07 |
| R² Score | 0.96 |

- **Visualization**:
  - A scatter plot of actual vs. predicted hypotenuse values is saved as `plots/actual_vs_predicted.png`.
  - The plot includes a red dashed line representing perfect prediction (`y = x`) for reference.
  - Generated using `matplotlib`.

## Observations

- **Performance**: The R² score of 0.96 indicates the model explains 96% of the variance in the hypotenuse, demonstrating a strong fit.
- **Error Analysis**: The RMSE of 6.07 suggests predictions are, on average, within 6.07 units of the true hypotenuse, reasonable given the range of `c` (up to \~141 for legs of 100).
- **Limitations**: Since Pythagoras' theorem is non-linear (`c = √(a² + b²)`), linear regression introduces minor errors, particularly for larger triangles, as seen in slight deviations in the scatter plot.
- **Insights**: This exercise shows how a simple linear model can approximate a non-linear mathematical relationship, useful for understanding machine learning applications in geometric contexts.

## Repository Structure

- `data/`
  - `triangle_dataset.csv`: Generated dataset of 1000 triangles.
- `src/`
  - `triangle_regression.py`: Python script implementing data generation, model training, evaluation, and plotting.
- `plots/`
  - `actual_vs_predicted.png`: Scatter plot of actual vs. predicted hypotenuse.
- `README.md`: This documentation file.

## How to Run

1. **Prerequisites**:

   - Python 3.8 or higher.
   - Install required libraries:

     ```bash
     pip install numpy pandas scikit-learn matplotlib
     ```

2. **Execute the Code**:

   - Run the script:

     ```bash
     python src/triangle_regression.py
     ```
   - Alternatively, open `src/triangle_regression.py` in a Jupyter Notebook if converted to `.ipynb`.

3. **Expected Outputs**:

   - **Dataset**: `data/triangle_dataset.csv`
   - **Plot**: `plots/actual_vs_predicted.png`
   - **Console Output**:

     ```
     Evaluation Metrics:
     MSE: 36.85
     RMSE: 6.07
     R²: 0.96
     ```

## Dependencies

- `numpy`: For random data generation and numerical operations.
- `pandas`: For dataset creation and CSV handling.
- `scikit-learn`: For linear regression and evaluation metrics.
- `matplotlib`: For plotting the actual vs. predicted visualization.

## Notes

- The implementation is optimized for simplicity and robustness, using a non-interactive Matplotlib backend (`Agg`) to avoid GUI-related issues.
- A warning about `Axes3D` may appear due to Matplotlib configurations but is suppressed and does not affect functionality, as 3D plotting is not required.
- The code is modular, with functions for data generation, training, evaluation, and plotting, inspired by functional programming principles.