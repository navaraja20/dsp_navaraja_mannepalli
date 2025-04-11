"""Inference module for house price prediction pipeline."""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Any
from joblib import load
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from .preprocess import preprocess_data

def load_artifacts() -> Tuple[Any, SimpleImputer,
                              StandardScaler, OneHotEncoder]:
    """Load persisted model,transformers from disk.
        Returns:
        Tuple containing (model, imputer, scaler, encoder)
        Raises:
        FileNotFoundError: If artifacts are missing
    """
    try:
        return (
            load('models/model.joblib'),
            load('models/imputer.joblib'),
            load('models/scaler.joblib'),
            load('models/encoder.joblib')
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Model artifacts not found. Train model first."
        ) from e
def make_predictions(
    input_data: pd.DataFrame,
    model: Optional[Any] = None,
    imputer: Optional[SimpleImputer] = None,
    scaler: Optional[StandardScaler] = None,
    encoder: Optional[OneHotEncoder] = None
) -> np.ndarray:
    """Generate predictions for new house data.
    Args:
        input_data: DataFrame containing raw features
        model: Optional pre-loaded model
        imputer: Optional pre-loaded imputer
        scaler: Optional pre-loaded scaler
        encoder: Optional pre-loaded encoder
    Returns:
        Numpy array of predicted house prices
    Raises:
        RuntimeError: If prediction fails
    """
    try:
        # Load artifacts if not provided
        if None in (model, imputer, scaler, encoder):
            model, imputer, scaler, encoder = load_artifacts()
        # Preprocess and predict
        processed_data, _, _, _ = preprocess_data(
            input_data, imputer, scaler, encoder
        )
        return model.predict(processed_data)
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}") from e
