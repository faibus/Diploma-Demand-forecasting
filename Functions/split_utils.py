import numpy as np
import pandas as pd

# Этот python файл имеет две функции
# filter_long_series()   - оставляет только те ряды, в которых достаточно данных для обучения
# split_train_val_test() - делит исходный датасет на три train + validation (1 окно) + test (3 окна)

def filter_long_series(
    real_demand: pd.DataFrame,
    horizon: int = 14,
    n_val_windows: int = 1,
    n_test_windows: int = 3,
    min_train_points: int = 35,
    id_col: str = "unique_id",
) -> pd.DataFrame:
    """
    Оставляем только ряды, у которых достаточно наблюдений
    для train + val + test.
    """
    min_required = min_train_points + horizon * (n_val_windows + n_test_windows)

    series_len = real_demand.groupby(id_col).size()
    long_series = series_len[series_len >= min_required].index

    return real_demand[real_demand[id_col].isin(long_series)].copy()


def split_train_val_test(
    real_demand_filtered: pd.DataFrame,
    horizon: int = 14,
    n_val_windows: int = 1,
    n_test_windows: int = 3,
    id_col: str = "unique_id",
    ds_col: str = "ds",
):
    """
    Делим данные на:
    - train: все даты, кроме последних holdout
    - val: первые n_val_windows * horizon дат из holdout
    - test: n_test_windows окон по horizon дат
    """
    df = real_demand_filtered.sort_values([id_col, ds_col]).reset_index(drop=True)

    all_dates = np.array(sorted(df[ds_col].unique()))
    required_holdout = horizon * (n_val_windows + n_test_windows)

    holdout_dates = all_dates[-required_holdout:]
    train_dates = all_dates[:-required_holdout]

    val_dates = holdout_dates[: horizon * n_val_windows]
    test_dates = holdout_dates[horizon * n_val_windows :]

    test_windows_dates = [
        test_dates[i * horizon : (i + 1) * horizon]
        for i in range(n_test_windows)
    ]

    train_df = df[df[ds_col].isin(train_dates)].copy()
    val_df = df[df[ds_col].isin(val_dates)].copy()
    test_windows_df = [df[df[ds_col].isin(w)].copy() for w in test_windows_dates]

    return train_df, val_df, test_windows_df