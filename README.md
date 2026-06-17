# Model Evaluation Dashboard

> Train, evaluate, visualize, and compare machine learning regression models from the command line.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## Overview

Model Evaluation Dashboard is a machine learning project built with Python and Scikit-Learn that allows users to:

- Load and validate datasets
- Select a target variable
- Train regression models
- Evaluate model performance
- Generate visual reports
- Compare multiple models
- Identify the best-performing model

This project was developed as part of my Machine Learning learning journey and focuses on understanding the complete model evaluation workflow.

---

## Features

### Data Validation

Before training, the dataset is automatically checked for:

- Missing values
- Duplicate rows
- Dataset validity

Example:

```text
=== DATA VALIDATION REPORT ===

Missing Values: 0
Duplicates: 0
Valid Dataset: True
```

### Single Model Evaluation

Train and evaluate one model at a time.

Available models:

| Option | Model |
|----------|----------|
| 1 | Linear Regression |
| 2 | Decision Tree Regressor |
| 3 | Random Forest Regressor |

### Model Comparison Mode

Evaluate all supported models automatically and compare their performance.

Example:

```text
=== MODEL COMPARISON ===

Model                MAE      RMSE      R²
--------------------------------------------------
Linear Regression    0.3473   0.4585   0.7617
Decision Tree        0.2517   0.3281   0.8780
Random Forest        0.1919   0.2531   0.9274

==================================================
Best Model: Random Forest
R² Score: 0.9274
```

### Evaluation Metrics

The project uses three common regression metrics:

| Metric | Description |
|----------|----------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² Score | Coefficient of Determination |

### Visualization Reports

The application automatically generates and saves:

#### Actual vs Predicted Plot

Shows how closely predictions match actual values.

```text
reports/actual_vs_predicted.png
```

#### Residual Plot

Helps identify prediction errors and model bias.

```text
reports/residual_plot.png
```

### Continuous Analysis Mode

Analyze multiple datasets without restarting the application.

Example:

```text
Analyze another dataset? (y/n)
```

---

## Project Structure

```text
model_evaluation_dashboard/
│
├── data/
│   └── processed_data.csv
│
├── reports/
│   ├── actual_vs_predicted.png
│   └── residual_plot.png
│
├── src/
│   ├── loader.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── visualizer.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Workflow

```text
Load Dataset
      │
      ▼
Validate Dataset
      │
      ▼
Select Target Column
      │
      ▼
Train/Test Split
      │
      ▼
Choose Mode
 ┌───────────────┐
 │ Single Model  │
 │ Compare Models│
 └───────────────┘
      │
      ▼
Train Models
      │
      ▼
Evaluate Performance
      │
      ▼
Generate Visualizations
      │
      ▼
Identify Best Model
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/aakhmadkhujaev/model-evaluation-dashboard.git
```

Move into the project:

```bash
cd model-evaluation-dashboard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python main.py
```

---

## Example Run

```text
Enter dataset name: processed_data.csv

Enter target column: mpg

Choose Mode:
1. Single Model
2. Compare Models

Enter choice: 2
```

Output:

```text
Best Model: Random Forest

R² Score: 0.9274
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Git
- GitHub

---

## Skills Demonstrated

This project demonstrates:

- Data Validation
- Train/Test Splitting
- Regression Modeling
- Model Evaluation
- Error Analysis
- Data Visualization
- Project Organization
- Git Branching Workflow
- Machine Learning Pipeline Design

---

## Future Improvements

Planned enhancements:

- Cross Validation
- Hyperparameter Tuning
- Feature Importance Analysis
- Classification Models
- Export Evaluation Reports
- Interactive Dashboard (Streamlit)

---

## Author

**Abror Ahmatadxujayev**

Software Engineering Student | Machine Learning Learner

GitHub: `aakhmadkhujaev`