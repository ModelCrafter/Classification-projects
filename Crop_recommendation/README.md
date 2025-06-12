# Crop Recommendation Classification

## Project Overview
This project implements a crop recommendation system using machine learning techniques. The primary goal is to classify crops based on given agricultural parameters such as soil pH, nitrogen, phosphorous, potassium levels, and other environmental factors.

The pipeline involves:
- Data loading and preparation.
- Data preprocessing, including feature engineering.
- Training a `RandomForestClassifier` model with hyperparameter optimization using GridSearchCV.
- Evaluating the model's performance on test data.

---

## Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `zipfile`.

Install dependencies using pip:
```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## Code Components

### 1. Data Loading
The `load_data` function loads the dataset from a ZIP archive and extracts it if not already done. The data is read into a Pandas DataFrame from the CSV file.

```python
def load_data() -> pd.DataFrame:
    # Load and extract the dataset
```

### 2. Data Splitting
The dataset is split into training (80%) and testing (20%) subsets:
- `X_train`, `Y_train`: Features and labels for training.
- `X_test`, `Y_test`: Features and labels for testing.

### 3. Feature Engineering
Feature engineering is performed using custom functions and a `ColumnTransformer`:
- **`determine_ph`**: Classifies pH values into acidic (-1), neutral (0), or alkaline (1).
- **`add_ph_type`**: Adds a `ph_type` column based on the pH classification.
- **`add_sum_all`**: Adds a `sum_all` column, which is the sum of all numeric columns.

### 4. Model Pipeline
A `Pipeline` integrates preprocessing and a `RandomForestClassifier`. The preprocessing step applies transformations using `StandardScaler` and feature engineering functions.

```python
finally_pip = Pipeline([
    ('processing', preprocess),
    ('classifier', RandomForestClassifier(random_state=42))
])
```

### 5. Hyperparameter Tuning
GridSearchCV is used to fine-tune the RandomForestClassifier. Hyperparameters tuned include:
- `max_features`
- `n_estimators`
- `max_depth`

### 6. Model Evaluation
The model is evaluated using:
- Accuracy
- F1 Score
- Precision
- Recall

---

## Results
- Without standardization: **94% accuracy**
- With standardization:
  - **96.88% accuracy** (with all columns)
  - **96.70% accuracy** (without all columns)
  - **97.15% accuracy** (with `ph_type` column and all columns)

---

## How to Run
1. Clone the repository.
2. Place the dataset archive (`archive.zip`) in `D:/python/datasets/`.
3. Execute the script:
   ```bash
   python crop_recommendation.py
   ```

---

## Key Notes
- Modify the `param_grid` dictionary to experiment with different hyperparameters.
- Ensure the dataset is correctly placed as per the directory structure.
- The pipeline and preprocessing steps are modular, allowing easy modification for other datasets.

---

## License
This project is licensed under the Apache License 2.0.

---

## Acknowledgments
Special thanks to the dataset providers and the scikit-learn library for simplifying machine learning workflows.

---
## Developer
***Youssef Khaled***
