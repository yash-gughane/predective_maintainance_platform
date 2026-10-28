def analyze_correlations(df):
    """
    Display correlations between numerical
    variables and RUL.
    """

    correlation = (
        df.corr(numeric_only=True)["RUL"]
        .sort_values()
    )

    print()
    print("=" * 50)
    print("CORRELATION WITH RUL")
    print("=" * 50)

    print(correlation)

    return correlation