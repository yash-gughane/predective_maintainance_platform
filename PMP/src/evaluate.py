from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def make_predictions(model, X_test):
    """
    Generate RUL predictions.
    """

    predictions = model.predict(
        X_test
    )

    return predictions


def evaluate_predictions(
    y_test,
    predictions
):
    """
    Calculate MAE, RMSE and R².
    """

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    return mae, rmse, r2