import pandas as pd
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def load_data(train_path: PathLike, test_path: PathLike):
    """
    Load train and test datasets.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def load_single_csv(path: PathLike):
    """
    Load a single CSV file.
    """
    return pd.read_csv(path)


def basic_cleaning(df: pd.DataFrame):
    """
    Drop unused identifier columns.
    """
    drop_cols = [c for c in ["id", "Unnamed: 0"] if c in df.columns]
    return df.drop(columns=drop_cols)
