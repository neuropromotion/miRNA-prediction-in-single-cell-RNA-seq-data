"""Inference package: miRPredictor + StackPredictor."""

try:
    from .mirpredictor import miRPredictor, SingleCell
    from .stack_predictor import StackPredictor
except ImportError:
    from mirpredictor import miRPredictor, SingleCell
    from stack_predictor import StackPredictor

__all__ = ["miRPredictor", "SingleCell", "StackPredictor"]
