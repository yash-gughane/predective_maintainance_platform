def add_rolling_mean(df, sensor, window=5):
    """
    Calculate rolling mean for a sensor
    separately for each engine.
    """

    feature_name = f"{sensor}_ma{window}"

    df[feature_name] = (
        df.groupby("engine_id")[sensor]
        .transform(
            lambda x: x.rolling(
                window=window,
                min_periods=1
            ).mean()
        )
    )

    return df


def add_rolling_std(df, sensor, window=5):
    """
    Calculate rolling standard deviation
    for a sensor separately for each engine.
    """

    feature_name = f"{sensor}_std{window}"

    df[feature_name] = (
        df.groupby("engine_id")[sensor]
        .transform(
            lambda x: x.rolling(
                window=window,
                min_periods=1
            ).std()
        )

    return df


def add_difference(df, sensor):
    """
    Calculate the change in a sensor value
    from the previous cycle.
    """

    feature_name = f"{sensor}_diff"

    df[feature_name] = (
        df.groupby("engine_id")[sensor]
        .diff()
    )

    return df