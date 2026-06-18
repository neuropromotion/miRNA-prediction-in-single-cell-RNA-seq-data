## Final Model

The final training, testing, and inference pipeline is based on the best-performing approach identified during model benchmarking.

### Architecture

The final ensemble consists of:

- CatBoost (hyperparameters optimized using Optuna)
- TabM
- ResNet-like neural network

### Ensembling Strategy

Predictions from the individual models are combined using a stacking framework with Ridge Regression as the meta-learner.
