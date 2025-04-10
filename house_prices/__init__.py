# Initialize as Python package
from .preprocess import get_feature_names
from .train import build_model
from .inference import make_predictions

__all__ = ['get_feature_names', 'build_model', 'make_predictions']
