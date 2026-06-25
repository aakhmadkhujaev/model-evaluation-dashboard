# 📊 Model Evaluation Dashboard v2.0

A professional machine learning dashboard for training, evaluating, visualizing, and saving regression and classification models.

This project provides an end-to-end workflow for preparing data, training machine learning models, evaluating their performance, generating reports, visualizing results, and saving trained models for future use.

---

# ✨ Features

## 📂 Dataset Management

* Load datasets from CSV files
* Validate dataset quality before training
* Check missing values
* Detect duplicate rows
* Display dataset information
* Select the target column interactively

---

## 🤖 Machine Learning Tasks

### Regression

* Linear Regression

### Classification

* Logistic Regression

---

## ⚙️ Data Preprocessing

* Automatic Train/Test Split
* Automatic feature encoding using `ColumnTransformer`
* One-Hot Encoding for categorical features
* Numerical feature handling
* Reusable preprocessing pipeline

---

## 📈 Model Evaluation

### Regression Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

### Classification Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Full Classification Report

---

## 📊 Visualizations

### Regression

* Actual vs Predicted Plot
* Residual Plot

### Classification

* Confusion Matrix

---

## 💾 Model Persistence

* Save trained regression models
* Save trained classification models
* Store models using Joblib (`.pkl`)

---

## 📝 Generated Reports

The dashboard automatically generates:

* Classification Report
* Evaluation Metrics
* Confusion Matrix
* Regression Visualizations
* Saved Machine Learning Models

---

# 📁 Project Structure

```text
model_evaluation_dashboard/
│
├── data/
│
├── outputs/
│   ├── figures/
│   │   ├── actual_vs_predicted.png
│   │   ├── residual_plot.png
│   │   └── confusion_matrix.png
│   │
│   ├── models/
│   │   ├── linear_regression.pkl
│   │   └── logistic_regression.pkl
│   │
│   └── reports/
│       └── classification_report.txt
│
├── src/
│   ├── loader.py
│   ├── validator.py
│   ├── preprocessor.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── visualizer.py
│   ├── model_saver.py
│   └── utils.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/aakhmadkhujaev/model-evaluation-dashboard.git
```

Navigate into the project

```bash
cd model-evaluation-dashboard
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

Run the application

```bash
python main.py
```

The dashboard will guide you through the following steps:

1. Load a dataset
2. Validate dataset quality
3. Select the target column
4. Choose a machine learning task
5. Train the model
6. Evaluate model performance
7. Generate reports and visualizations
8. Save the trained model
9. Analyze another dataset or exit

---

# 📦 Outputs

Generated files are stored inside the `outputs` directory.

## Figures

* Actual vs Predicted Plot
* Residual Plot
* Confusion Matrix

## Reports

* Classification Report

## Models

* Linear Regression Model
* Logistic Regression Model

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib

---

# 📌 Future Improvements

* Support multiple regression models
* Support multiple classification models
* Automatic best model selection
* ROC Curve visualization
* Precision–Recall Curve
* Feature Importance
* SHAP Explainability
* Complete Scikit-learn Pipeline Persistence

---

# 👨‍💻 Author

**Abror Ahmadxujaev**

Software Engineering Student

Turin Polytechnic University in Tashkent

---

# ⭐ If you found this project useful, consider giving it a star.
