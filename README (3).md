# Glass Classifier Project

This project focuses on classifying different types of glass using machine learning and deep learning techniques. Below is a comprehensive overview of the project structure, requirements, and implementation details.

## Project Overview

The Glass Classifier aims to predict the type of glass based on its chemical composition. This involves analyzing features such as refractive index and the presence of various chemical elements (e.g., Sodium, Magnesium, Aluminum) in the glass. Applications include material science, archaeology, and forensics, where identifying glass types is critical.

## Dataset Details

The dataset used in this project is the **Glass Identification Dataset** from the UCI Machine Learning Repository. It contains:

- **Features:** 9 attributes, including:
  - Refractive Index
  - Sodium (Na)
  - Magnesium (Mg)
  - Aluminum (Al)
  - Silicon (Si)
  - Potassium (K)
  - Calcium (Ca)
  - Barium (Ba)
  - Iron (Fe)
- **Target Variable:** Glass type (1-7), where each number represents a specific type of glass (e.g., building windows, containers, etc.).

The dataset includes 214 samples.

## Implementation Steps

### 1. Data Preprocessing

- **Handling Missing Values:** The dataset has no missing values, simplifying preprocessing.
- **Feature Scaling:** Standardized features using `Stander scaler` normalization to ensure uniform model performance.
- **Train-Test Split:** Divided the dataset into training (80%) and testing (20%) sets.

### 2. Exploratory Data Analysis (EDA)

- **Correlation Analysis:** Visualized feature correlations using a heatmap to identify relationships between attributes.
- **Feature Engineering:** Added new columns derived from existing features to improve accuracy and explored their correlation with the target variable.
- **Visualizations:** Created histograms to show the distribution of the target variable and the new features.

### 3. Machine Learning Models

- **Random Forest Classifier:**
  - Used default hyperparameters without tuning.
  - Achieved an accuracy of \~80% on the test set.
- **Evaluation Metric:** Only accuracy was used to evaluate performance.

### 4. Deep Learning Approach

- **Model Architecture:**
  - Input Layer: 9 neurons (one per feature).
  - Hidden Layers: Two dense layers with 128 and 64 neurons, using ReLU activation.
  - Output Layer: 7 neurons (one for each glass type) with softmax activation.
- **Training Details:**
  - Optimizer: Adam with a learning rate of 0.001.
  - Loss Function: Categorical Crossentropy for multi-class classification.
  - Batch Size: 32, Epochs: 100.
  - Early Stopping: Monitored validation loss to prevent overfitting.
- **Results:** Achieved \~85% accuracy on the test set, demonstrating strong performance compared to machine learning models.

### 5. Visualizations

- **Training Progress:** Plotted accuracy and loss curves for each epoch to monitor the deep learning model.
- **Confusion Matrices:** Generated for both Random Forest and Neural Network to evaluate classification performance.

## Requirements

Ensure you have the following installed:

- Python 3.8+
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow (for deep learning)

Install required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## File Structure

- `Glass_classifier.ipynb`: The main notebook containing the entire workflow, from data preprocessing to model evaluation.
- `README.md`: This file, providing an overview of the project.
- `data/`: Directory containing the dataset (CSV format).
- `models/`: Directory for saving trained models.
- `utils/`: Directory with helper functions for preprocessing and visualization.

## How to Run

1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd glass-classifier
   ```
3. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook Glass_classifier.ipynb
   ```
4. Execute the notebook step-by-step to reproduce the results.

## Results

### Machine Learning Models

- **Random Forest Classifier:** Achieved an accuracy of \~80%, with strong performance on majority classes.

### Deep Learning Model

- **Neural Network:** Outperformed traditional models with \~85% accuracy, demonstrating robustness in handling complex patterns.

### Key Insights

- **Important Features:** Sodium, Calcium, and Refractive Index were critical for classification.
- **Challenges:** Overlap between certain glass types led to occasional misclassifications, particularly between Type 1 and Type 2.

## Future Directions

- Experiment with advanced neural network architectures, such as CNNs, to extract more features.
- Implement automated hyperparameter tuning using frameworks like Optuna.
- Develop a web interface for live predictions.

## Acknowledgments

- [Glass Dataset](https://archive.ics.uci.edu/ml/datasets/glass+identification): Sourced from the UCI Machine Learning Repository.

## License

This project is licensed under the MIT License. Feel free to use and modify it as needed.

