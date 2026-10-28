from sklearn.model_selection import train_test_split


def split_by_engine(
    df,
    train_ratio=0.7,
    random_state=42
):
    """
    Split dataset by engine ID.

    This ensures that engines in the test set
    are completely unseen during training.
    """

    engine_ids = df["engine_id"].unique()

    train_engines, test_engines = train_test_split(
        engine_ids,
        train_size=train_ratio,
        random_state=random_state
    )

    train_df = df[
        df["engine_id"].isin(train_engines)
    ].copy()

    test_df = df[
        df["engine_id"].isin(test_engines)
    ].copy()

    X_train = train_df.drop(
        columns=["RUL"]
    )

    y_train = train_df["RUL"]

    X_test = test_df.drop(
        columns=["RUL"]
    )

    y_test = test_df["RUL"]

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )