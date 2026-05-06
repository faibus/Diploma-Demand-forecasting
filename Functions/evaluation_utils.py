import pandas as pd


def evaluate_frozen_windows(
    predict_one_window_fn,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_windows_df: list,
    horizon: int = 14,
    ds_col: str = "ds",
):
    """
    Оценка в режиме frozen weights:
    - predict на val по train
    - predict на test-окнах с расширением history фактами предыдущих окон
    - без дообучения модели внутри функции
    """

    # VAL
    val_pred = predict_one_window_fn(
        history_df=train_df,
        target_window_df=val_df,
        horizon=horizon,
    )
    val_pred["cutoff"] = val_df[ds_col].min() - pd.Timedelta(days=1)

    # TEST
    history_df = pd.concat([train_df, val_df], ignore_index=True)
    test_preds = []

    for i, win_df in enumerate(test_windows_df, 1):
        pred_i = predict_one_window_fn(
            history_df=history_df,
            target_window_df=win_df,
            horizon=horizon,
        )
        pred_i["cutoff"] = win_df[ds_col].min() - pd.Timedelta(days=1)
        pred_i["test_window"] = i
        test_preds.append(pred_i)

        # Добавляем факты окна в историю (контекст), но модель не переобучаем
        history_df = pd.concat([history_df, win_df], ignore_index=True)

    test_pred = pd.concat(test_preds, ignore_index=True)
    return val_pred, test_pred