import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_feature_names() -> tuple[list, list]:
    """Return continuous and categorical feature names."""
    continuous_features = ['LotArea', 'GrLivArea']
    categorical_features = ['MSZoning', 'Neighborhood']
    return continuous_features, categorical_features

def preprocess_data(
    df: pd.DataFrame,
    imputer: SimpleImputer = None,
    scaler: StandardScaler = None,
    encoder: OneHotEncoder = None,
    fit: bool = False
) -> np.ndarray:
    """
    Preprocess data with imputation, scaling, and encoding.
    If fit=True, fits transformers. Otherwise, uses pre-fitted transformers.
    """
    cont_feats, cat_feats = get_feature_names()
    
    # Impute categorical
    if fit:
        imputer = SimpleImputer(strategy='most_frequent')
        imputed_cat = imputer.fit_transform(df[cat_feats])
    else:
        imputed_cat = imputer.transform(df[cat_feats])
    
    # Scale continuous
    if fit:
        scaler = StandardScaler()
        scaled_cont = scaler.fit_transform(df[cont_feats])
    else:
        scaled_cont = scaler.transform(df[cont_feats])
    
    # Encode categorical
    if fit:
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoded_cat = encoder.fit_transform(imputed_cat).toarray()
    else:
        encoded_cat = encoder.transform(imputed_cat).toarray()
    
    return np.concatenate([scaled_cont, encoded_cat], axis=1), imputer, scaler, encoder