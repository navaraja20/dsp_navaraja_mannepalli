import numpy as np
import pandas as pd
from joblib import load
from .preprocess import preprocess_data, get_feature_names

def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """Load persisted objects and make predictions on new data."""
    # Load objects
    model = load('models/model.joblib')
    imputer = load('models/imputer.joblib')
    scaler = load('models/scaler.joblib')
    encoder = load('models/encoder.joblib')
    
    # Preprocess and predict
    processed_data, _, _, _ = preprocess_data(input_data, imputer, scaler, encoder)
    return model.predict(processed_data)