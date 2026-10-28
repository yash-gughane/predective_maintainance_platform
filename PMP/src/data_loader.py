import pandas as pd


COLUMN_NAMES = [
    "engine_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3"
]

COLUMN_NAMES += [
    f"sensor{i}"
    for i in range(1, 22)
]


def load_dataset(path):
    """
    Load NASA C-MAPSS dataset.
    """

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None
    )

    df.columns = COLUMN_NAMES

    return df


def validate_dataset(df):
    """
    Print basic information about the dataset.
    """

    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)

    print()
    print(f"Shape : {df.shape}")

    print()
    print(
        f"Missing Values : "
        f"{df.isnull().sum().sum()}"
    )

    print()
    print(
        f"Duplicate Rows : "
        f"{df.duplicated().sum()}"
    )

    print()
    print("Data Types")
    print(df.dtypes)

    print()
    print("Statistical Summary")
    print(df.describe())


def create_rul(df):
    """
    Calculate Remaining Useful Life (RUL)
    for every observation.

    RUL = maximum cycle of that engine - current cycle
    """

    max_cycles = (
        df.groupby("engine_id")["cycle"]
        .max()
        .reset_index()
    )

    max_cycles.columns = [
        "engine_id",
        "max_cycle"
    ]

    df = df.merge(
        max_cycles,
        on="engine_id",
        how="left"
    )

    df["RUL"] = (
        df["max_cycle"]
        - df["cycle"]
    )

    df = df.drop(
        columns=["max_cycle"]
    )

    return df