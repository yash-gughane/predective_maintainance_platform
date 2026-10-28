from sklearn.ensemble import RandomForestRegressor


def train_random_forest(
    X_train,
    y_train
):
    """
    Train the baseline Random Forest model.
    """

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model