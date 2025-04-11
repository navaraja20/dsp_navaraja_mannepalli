"""Model training module for house price prediction pipeline."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from .preprocess import preprocess_data

def build_model(data: pd.DataFrame) -> Dict[str, float]:
    """Build and evaluate house price prediction model.
    Args:
        data: Training DataFrame containing both features and 'SalePrice'target
    Returns:
        Dictionary with RMSLE metric: {'rmsle': float}
    Raises:
        RuntimeError: If model training fails
    """
    try:
        # Create models directory if not exists
        Path('models').mkdir(exist_ok=True)
        # Split data
        X = data.drop(columns=['SalePrice'])
        y = data['SalePrice']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Preprocess with fitting
        X_train_processed, imputer, scaler, encoder = preprocess_data(
            X_train, fit=True
        )
        # Train model
        model = LinearRegression()
        model.fit(X_train_processed, y_train)
        # Evaluate
        X_test_processed, _, _, _ = preprocess_data(
            X_test, imputer, scaler, encoder
        )
        y_pred = model.predict(X_test_processed)
        rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
        # Persist objects
        dump(model, 'models/model.joblib')
        dump(imputer, 'models/imputer.joblib')
        dump(scaler, 'models/scaler.joblib')
        dump(encoder, 'models/encoder.joblib')
        return {'rmsle': round(rmsle, 4)}
    except Exception as e:
        raise RuntimeError(f"Model training failed: {str(e)}") from e
