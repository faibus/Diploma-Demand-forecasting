import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

############################################################################################## 
# Функция собирает все прогнозы различных моделей в один общий датасет для расчёта ошибок    #
# Сбор прогнозов идёт в рамках составного ключа (товар + прогнозируемая дата + отсечка окна) #
##############################################################################################

KEYS = ["unique_id", "ds", "cutoff"]

def intersect_cv_frames(
    arima_exog: pd.DataFrame,
    ml: pd.DataFrame,
    nl: pd.DataFrame,
    smooth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Внутреннее пересечение по (unique_id, ds, cutoff).
    y берём из arima_exog.
    """
    # база
    merged = arima_exog[KEYS + ["y"]].drop_duplicates(KEYS)

    # прогнозы ARIMA
    arima_preds = [c for c in arima_exog.columns if c not in KEYS and c != "y"]
    merged = merged.merge(
        arima_exog[KEYS + arima_preds].drop_duplicates(KEYS),
        on=KEYS,
        how="inner",
    )

    # прогнозы MLForecast
    ml_preds = [c for c in ml.columns if c not in KEYS and c != "y"]
    merged = merged.merge(
        ml[KEYS + ml_preds].drop_duplicates(KEYS),
        on=KEYS,
        how="inner",
    )

    # прогнозы NeuralForecast
    nl_preds = [c for c in nl.columns if c not in KEYS and c != "y"]
    merged = merged.merge(
        nl[KEYS + nl_preds].drop_duplicates(KEYS),
        on=KEYS,
        how="inner",
    )

    # прогнозы smooth/stat блока
    smooth_preds = [c for c in smooth.columns if c not in KEYS and c != "y"]
    merged = merged.merge(
        smooth[KEYS + smooth_preds].drop_duplicates(KEYS),
        on=KEYS,
        how="inner",
    )

    return merged



################################################ 
# Отрисовка ошибок прогнозов различных моделей #
################################################
def model_color(name):
    name = str(name)
    # Нейронки
    if name in ("LSTM", "GRU", "NHITS"):
        return "#5B4B8A"
    # ML
    if name == "lgb_mae_base" or name.startswith("lgb"):
        return "#E9C46A"
    # ARIMA_exog
    if name == "AutoARIMA_exog":
        return "#E76F51"
    # Всё остальное = классические статистические модели
    return "#2A9D8F"

def plot_metrics_bars(
    summary_mean,
    metrics=("mae", "rmse", "smape", "wape"),
    layout=(2, 2),
    figsize=(14, 12),
):
    """
    Рисует barplot по метрикам из summary_mean.
    layout=(rows, cols), например:
    - (2, 2) для 4 метрик
    - (2, 1) для 2 метрик
    """
    rows, cols = layout
    if len(metrics) > rows * cols:
        raise ValueError("Количество метрик больше количества ячеек в layout.")
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Приводим axes к плоскому списку для единообразия
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.ravel()
    for i, col_name in enumerate(metrics):
        ax = axes[i]
        s = summary_mean[col_name].sort_values()
        bars = ax.bar(
            s.index.astype(str),
            s.values,
            color=[model_color(m) for m in s.index]
        )
        ax.set_ylabel(col_name.upper())
        ax.tick_params(axis="x", rotation=90)
        
        # Подписи над столбцами (с учетом масштаба)
        y_max = s.max() if len(s) else 0
        offset = y_max * 0.02 if y_max > 0 else 0.02
        ax.set_ylim(top=y_max * 1.12 if y_max > 0 else 1)
        for bar, val in zip(bars, s.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + offset,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    # Если ячеек больше, чем метрик — скрываем лишние
    for j in range(len(metrics), rows * cols):
        axes[j].axis("off")
    handles = [
        Patch(facecolor="#5B4B8A", label="Нейросети"),
        Patch(facecolor="#2A9D8F", label="Стат. модели"),
        Patch(facecolor="#E9C46A", label="ML"),
        Patch(facecolor="#E76F51", label="ARIMA_exog"),
    ]
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()
    
    
    
#######################################################
# Функции для расчета MAE и WAPE в денежном выражении #
#######################################################

def money_mae_wape(df: pd.DataFrame, model: str, y_col: str = "y", p_col: str = "price"):
    y = df[y_col].to_numpy(dtype=float)
    p = df[p_col].to_numpy(dtype=float)
    yhat = df[model].to_numpy(dtype=float)
    
    err_money = np.abs(p * (y - yhat))
    denom = np.abs(p * y).sum()
    
    mae_money = err_money.mean()
    wape_money = np.nan if denom == 0 else err_money.sum() / denom * 100.0
    
    return mae_money, wape_money

def summarize_money_metrics(
    cv_results: pd.DataFrame,
    price_table: pd.DataFrame,
    id_col: str = "unique_id",
    y_col: str = "y",
    price_col: str = "price",
    extra_id_cols: tuple = ("ds", "cutoff"),
):
    """
    Джойнит price_table к cv_results и считает money-MAE / money-WAPE по всем моделям.
    """
    money_cv = pd.merge(cv_results, price_table, how="left", on=id_col)

    id_cols = [id_col, y_col, price_col] + list(extra_id_cols)
    model_cols = [c for c in money_cv.columns if c not in id_cols]

    rows = []
    for m in model_cols:
        mae_m, wape_m = money_mae_wape(money_cv, m, y_col=y_col, p_col=price_col)
        rows.append({"model": m, "mae_money": mae_m, "wape_money": wape_m})

    summary_money = pd.DataFrame(rows).sort_values("mae_money").set_index("model")
    return summary_money, money_cv