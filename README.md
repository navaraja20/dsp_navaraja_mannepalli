# 🏠 House Price Prediction Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python) ![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-blue?style=for-the-badge&logo=numpy) ![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-blue?style=for-the-badge&logo=pandas) ![scikit-learn](https://img.shields.io/badge/Scikit%20Learn-1.0%2B-orange?style=for-the-badge&logo=scikit-learn) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter) ![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## 📌 Overview

A comprehensive **machine learning pipeline** for predicting house prices using **linear regression**, featuring full industrialization from exploratory analysis to production-ready deployment. This project demonstrates end-to-end ML best practices including data preprocessing, model training, hyperparameter optimization, and inference.

## 🎯 Key Features

- 📊 **End-to-End Pipeline** - Data preprocessing → Model training → Inference
- 🏭 **Production Ready** - Persisted models and transformers for deployment
- 🔧 **Modular Design** - Separated preprocessing, training, and inference modules
- ✅ **Reproducible** - Type hints, docstrings, and linting enforcement (Flake8)
- 📦 **Packaged** - Installable Python package structure
- 📈 **High Performance** - Optimized linear regression with feature engineering
- 🔍 **Comprehensive** - Includes EDA notebooks and visualizations
- ⚙️ **Scalable** - Ready for batch predictions and integration

## 📦 Technologies Used

- **Python 3.8+** - Programming language
- **NumPy 1.21+** - Numerical computing
- **Pandas 1.3+** - Data manipulation and analysis
- **Scikit-learn 1.0+** - Machine learning models and preprocessing
- **Joblib** - Model serialization
- **Flake8** - Code linting and quality
- **Jupyter Notebook** - Interactive development

## 📊 Dataset

[Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset with:

- **79 explanatory variables** describing residential properties
- **Continuous and categorical features** with missing values
- **Target variable:** `SalePrice` (regression task)
- **Training set:** 1,460 samples
- **Test set:** 1,459 samples

## 📂 Project Structure

```
dsp_navaraja_mannepalli/
├── house_prices/                  # Main package
│   ├── __init__.py
│   ├── preprocessing.py          # Data preprocessing utilities
│   ├── training.py               # Model training logic
│   ├── inference.py              # Prediction functionality
│   └── utils.py                  # Helper functions
├── notebooks/                    # Jupyter notebooks
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   └── 02_modeling.ipynb        # Model development
├── models/                       # Saved models directory
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── pyproject.toml               # Project configuration
├── test.py                      # Unit tests
└── README.md                    # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone Repository

```bash
git clone https://github.com/navaraja20/dsp_navaraja_mannepalli.git
cd dsp_navaraja_mannepalli
```

### Step 2: Create Virtual Environment

```bash
# Using conda
conda create -n house-prices python=3.8
conda activate house-prices

# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Package

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## 💻 Usage

### Training a Model

```python
from house_prices import build_model
import pandas as pd

# Load training data
train_df = pd.read_csv("data/train.csv")

# Build and train model
results = build_model(train_df)
print(f"Model RMSLE: {results['rmsle']:.4f}")
```

### Making Predictions

```python
from house_prices import make_predictions

# Load test data
test_df = pd.read_csv("data/test.csv")

# Generate predictions
predictions = make_predictions(test_df)
print(predictions.head())
```

### Notebook Usage

Explore the project interactively:

```bash
jupyter notebook notebooks/
```

## 🔧 Development

### Install Development Dependencies

```bash
pip install flake8 autopep8 pytest
```

### Code Linting

```bash
# Check code style
flake8 house_prices/

# Auto-format code
autopep8 --in-place --aggressive house_prices/*.py
```

### Running Tests

```bash
python test.py
```

## 📈 Model Performance

- **Algorithm:** Linear Regression with feature engineering
- **Evaluation Metric:** Root Mean Squared Logarithmic Error (RMSLE)
- **Cross-validation:** k-fold validation
- **Optimization:** Hyperparameter tuning with grid search

## 🤝 Contributing

Contributions are welcome! Please feel free to:

- Submit issues for bugs or suggestions
- Fork the repository and submit pull requests
- Improve documentation and examples
- Add new features or models
- Optimize existing code

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👤 Author

**Navaraja Mannepalli**
- 🔗 GitHub: [@navaraja20](https://github.com/navaraja20)
- 💼 LinkedIn: [Navaraja Mannepalli](https://linkedin.com/in/navaraja-mannepalli)
- 📧 Email: navaraja13@gmail.com

## 📚 References

- [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Feature Engineering Best Practices](https://developers.google.com/machine-learning/crash-course/representation/feature-engineering)

---

⭐ If you found this helpful, please consider giving it a star!

📌 **Last Updated:** November 2024
