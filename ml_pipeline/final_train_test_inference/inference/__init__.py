"""Inference package: SingleCell preprocessor + StackPredictor."""

try:
    from .preprocessor import SingleCell
    from .stack_predictor import StackPredictor
except ImportError:
    from preprocessor import SingleCell
    from stack_predictor import StackPredictor

__all__ = ["SingleCell", "StackPredictor"]
