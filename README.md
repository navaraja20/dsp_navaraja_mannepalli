# House Price Prediction Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![Flake8](https://img.shields.io/badge/Flake8-5.0%2B-yellow)
![Joblib](https://img.shields.io/badge/Joblib-1.0%2B-green)

A machine learning pipeline for predicting house prices using linear regression, featuring full industrialization from exploratory analysis to production-ready deployment.

## 📌 Features

- **End-to-End Pipeline**: Data preprocessing → Model training → Inference
- **Production Ready**: Persisted models and transformers
- **Modular Design**: Separated preprocessing, training, and inference
- **Reproducible**: Type hints, docstrings, and linting enforcement
- **Packaged**: Installable Python package structure

## 🏠 Dataset

Using the [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset with:
- 79 explanatory variables
- Continuous and categorical features
- Target variable: `SalePrice`

## 🛠️ Installation

1. Clone the repository:
bash:
git clone https://github.com/yourusername/dsp_navaraja_mannepalli.git
cd dsp_navaraja_mannepalli

Create and activate virtual environment:
conda create 
conda activate    # Windows

Install dependencies:
pip install -r requirements.txt

Install package in development mode:
pip install -e .

🚀 Usage

from house_prices import build_model
import pandas as pd

train_df = pd.read_csv("data/train.csv")
results = build_model(train_df)
print(f"Model RMSLE: {results['rmsle']:.4f}")

Making Predictions:

from house_prices import make_predictions

test_df = pd.read_csv("data/test.csv")
predictions = make_predictions(test_df)

📂 Project Structure:

![image](https://github.com/user-attachments/assets/cc0ab29f-5671-412a-8645-a2a2af6aafea)

🔧 Development:

Install development requirements: pip install flake8 autopep8
Run linting: flake8 house_prices/
Format code: autopep8 --in-place --aggressive house_prices/*.py

📊 Results:

Metric	Value
RMSLE	0.210
R²	0.825

📧 Contact:
Navaraja Mannepalli - nava-raja.mannepalli@epita.fr

Project Link: https://github.com/yourusername/dsp_navaraja_mannepalli
