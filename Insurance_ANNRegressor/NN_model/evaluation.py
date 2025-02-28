import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_log_error,
)


def evaluate_model_performance(model, X_test, y_test, is_log_transformed=True):
    """
    Evaluates the performance of a trained model on the test dataset using multiple metrics.

    Args:
        model (torch.nn.Module): The trained neural network model.
        X_test (torch.Tensor): The input features for the test set.
        y_test (torch.Tensor): The actual target values for the test set.
        is_log_transformed (bool): Whether the target variable is log-transformed. Default is True.

    Returns:
        None
    """
    # Predict using the model
    y_test_pred = model(X_test)

    # If the target values were log-transformed, reverse the transformation
    if is_log_transformed:
        y_test_pred_actual = torch.exp(y_test_pred) - 1
        y_test_actual = torch.exp(y_test) - 1
    else:
        y_test_pred_actual = y_test_pred
        y_test_actual = y_test

    # Detach tensors and convert them to numpy arrays for evaluation
    y_test_pred_actual = y_test_pred_actual.detach().numpy()
    y_test_actual = y_test_actual.detach().numpy()

    # Calculate performance metrics
    rmsle = root_mean_squared_log_error(y_test_actual, y_test_pred_actual) ** 0.5
    mse = mean_squared_error(y_test_actual, y_test_pred_actual) ** 0.5
    mae = mean_absolute_error(y_test_actual, y_test_pred_actual)

    # Print results
    print(f" RMSLE: {rmsle:.4f}    | MSE: {mse:.4f}   | MAE {mae:.4f}")
    test_results = {"RMSLE": rmsle, "MSE": mse, "MAE": mae}
    return test_results


def test_data_to_tensor(X, y):
    X = torch.tensor(X.to_numpy(), dtype=torch.float32)
    y = torch.tensor(y.to_numpy(), dtype=torch.float32).unsqueeze(1)

    y = torch.log(y + 1)

    return X, y
