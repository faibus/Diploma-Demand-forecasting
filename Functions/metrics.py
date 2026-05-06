import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Служебные колонки в результатах CV, которые не являются прогнозами моделей
DEFAULT_RESERVED_COLUMNS = ("unique_id", "ds", "cutoff", "y")


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error, %."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator > 0
    if mask.sum() == 0:
        return np.nan
    return (2 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]).mean() * 100


def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error, %."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    denominator = np.abs(y_true).sum()
    if denominator == 0:
        return np.nan
    return np.abs(y_true - y_pred).sum() / denominator * 100


DEFAULT_METRICS = {
    "mae": mae,
    "rmse": rmse,
    "smape": smape,
    "wape": wape,
}


def get_model_columns(cv_results: pd.DataFrame, reserved_columns=DEFAULT_RESERVED_COLUMNS):
    """
    Возвращает список колонок с прогнозами моделей.
    """
    return [col for col in cv_results.columns if col not in reserved_columns]


def compute_metrics_per_window(
    cv_results: pd.DataFrame,
    model_columns,
    metrics_dict=None,
    cutoff_col: str = "cutoff",
    target_col: str = "y",
):
    """
    Считает метрики по каждому окну (cutoff) и по каждой модели.

    Parameters
    ----------
    cv_results : pd.DataFrame
        Результат cross_validation.
    model_columns : list[str]
        Имена колонок с прогнозами моделей.
    metrics_dict : dict[str, callable], optional
        Словарь метрик вида {'mae': mae, ...}. По умолчанию DEFAULT_METRICS.
    cutoff_col : str
        Название колонки окна CV.
    target_col : str
        Название колонки целевой переменной.

    Returns
    -------
    pd.DataFrame
        Колонки: cutoff, model, <метрики>.
    """
    if metrics_dict is None:
        metrics_dict = DEFAULT_METRICS

    records = []
    grouped = cv_results.groupby(cutoff_col)

    for cutoff, group in grouped:
        y_true = group[target_col].values
        for model in model_columns:
            y_pred = group[model].values
            row = {"cutoff": cutoff, "model": model}
            for metric_name, metric_func in metrics_dict.items():
                row[metric_name] = metric_func(y_true, y_pred)
            records.append(row)

    return pd.DataFrame(records)


def summarize_metrics(metrics_per_window: pd.DataFrame, metric_names=None):
    """
    Возвращает:
    - summary_mean: средние значения метрик по окнам
    - summary_stats: mean/std/min/max по окнам
    """
    if metric_names is None:
        metric_names = list(DEFAULT_METRICS.keys())

    metric_names = list(metric_names)
    summary_mean = metrics_per_window.groupby("model")[metric_names].mean()
    summary_stats = metrics_per_window.groupby("model")[metric_names].agg(["mean", "std", "min", "max"])
    return summary_mean, summary_stats


def build_volume_segments(
    demand_df: pd.DataFrame,
    id_col: str = "unique_id",
    target_col: str = "y",
    low_q: float = 0.25,
    high_q: float = 0.75,
):
    """
    Формирует сегменты SKU по среднему спросу.
    Возвращает dict: {segment_name: Index(unique_id)}.
    """
    avg_demand = demand_df.groupby(id_col)[target_col].mean()
    low_border = avg_demand.quantile(low_q)
    high_border = avg_demand.quantile(high_q)
    segments = {
        f"High volume (top {int((1 - high_q) * 100)}%)": avg_demand[avg_demand >= high_border].index,
        f"Mid volume ({int(low_q * 100)}-{int(high_q * 100)}%)": avg_demand[
            (avg_demand >= low_border) & (avg_demand < high_border)
        ].index,
        f"Low volume (bottom {int(low_q * 100)}%)": avg_demand[avg_demand < low_border].index,
    }
    return segments


def summarize_metrics_by_segments(
    cv_results: pd.DataFrame,
    demand_df_for_segmentation: pd.DataFrame,
    model_columns=None,
    metrics_dict=None,
    id_col: str = "unique_id",
    target_col: str = "y",
    low_q: float = 0.25,
    high_q: float = 0.75,
):
    """
    Считает метрики по сегментам объема SKU (High/Mid/Low).
    Возвращает DataFrame с MultiIndex:
    (segment, n_series, model) и колонками метрик.
    """
    if metrics_dict is None:
        metrics_dict = DEFAULT_METRICS
    if model_columns is None:
        model_columns = get_model_columns(cv_results)
    segments = build_volume_segments(
        demand_df=demand_df_for_segmentation,
        id_col=id_col,
        target_col=target_col,
        low_q=low_q,
        high_q=high_q,
    )
    segment_results = []
    metric_names = list(metrics_dict.keys())
    for segment_name, segment_ids in segments.items():
        mask = cv_results[id_col].isin(segment_ids)
        cv_segment = cv_results.loc[mask]
        # Если в сегменте по какой-то причине нет наблюдений в cv_results
        if cv_segment.empty:
            continue
        segment_metrics = compute_metrics_per_window(
            cv_results=cv_segment,
            model_columns=model_columns,
            metrics_dict=metrics_dict,
            cutoff_col="cutoff",
            target_col=target_col,
        )
        segment_summary = segment_metrics.groupby("model")[metric_names].mean()
        segment_summary["segment"] = segment_name
        segment_summary["n_series"] = len(segment_ids)
        segment_results.append(segment_summary)
    if not segment_results:
        return pd.DataFrame(columns=["segment", "n_series", "model"] + metric_names)
    summary_by_segment = pd.concat(segment_results)
    summary_by_segment = summary_by_segment.reset_index().set_index(["segment", "n_series", "model"])
    summary_by_segment = summary_by_segment.sort_index()
    return summary_by_segment