"""Data preprocessing module for house price prediction pipeline."""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_feature_names() -> Tuple[list[str], list[str]]:
    """Get feature names for continuous and categorical features.
    Returns:
        Tuple[list[str], list[str]]:
            First element contains continuous feature names,
            Second element contains categorical feature names
    """
    continuous_features = ['LotArea', 'GrLivArea']
    categorical_features = ['MSZoning', 'Neighborhood']
    return continuous_features, categorical_features

def preprocess_data(
    df: pd.DataFrame,
    imputer: Optional[SimpleImputer] = None,
    scaler: Optional[StandardScaler] = None,
    encoder: Optional[OneHotEncoder] = None,
    fit: bool = False
) -> Tuple[np.ndarray, SimpleImputer, StandardScaler, OneHotEncoder]:
    """Preprocess input data with imputation, scaling and encoding.
    Args:
        df: Input DataFrame containing raw features
        imputer: Optional pre-fitted SimpleImputer
        scaler: Optional pre-fitted StandardScaler
        encoder: Optional pre-fitted OneHotEncoder
        fit: Whether to fit new transformers
    Returns:
        Tuple containing:
            - Processed features as numpy array
            - Fitted imputer
            - Fitted scaler
            - Fitted encoder
    Raises:
        ValueError: If preprocessing fails
    """
    cont_feats, cat_feats = get_feature_names()
    try:
        # Impute categorical
        if fit or imputer is None:
            imputer = SimpleImputer(strategy='most_frequent')
            imputed_cat = imputer.fit_transform(df[cat_feats])
        else:
            imputed_cat = imputer.transform(df[cat_feats])
        # Scale continuous
        if fit or scaler is None:
            scaler = StandardScaler()
            scaled_cont = scaler.fit_transform(df[cont_feats])
        else:
            scaled_cont = scaler.transform(df[cont_feats])
        # Encode categorical
        if fit or encoder is None:
            encoder = OneHotEncoder(handle_unknown='ignore',
                                    sparse_output=False)
            encoded_cat = encoder.fit_transform(imputed_cat)
        else:
            encoded_cat = encoder.transform(imputed_cat)
        return (
            np.concatenate([scaled_cont, encoded_cat], axis=1),
            imputer,
            scaler,
            encoder
        )
    except Exception as e:
        raise ValueError(f"Preprocessing failed: {str(e)}") from e
