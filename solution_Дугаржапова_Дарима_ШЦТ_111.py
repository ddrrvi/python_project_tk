'''
Запуск:
python solution_Дугаржапова_Дарима_ШЦТ_111.py --data_dir ./data_train --output_dir ./output
'''
import os
import argparse
import warnings
import glob
import decimal

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# ПАРАМЕТРЫ АЛГОРИТМА (статистически обоснованы, не подогнаны)
# ═══════════════════════════════════════════════════════════════
RZSCORE_THRESHOLD = 3.5   # Robust Z-Score: 3.5σ — экстремальный выброс (<0.02% хвоста)
SHARE_THRESHOLD   = 0.50  # Доля ≥50% в OTS бренда за день — аномальная концентрация
IQR_FACTOR        = 3.0   # Tukey fence: Q75 + 3·IQR (расширенный, только правый хвост)
MIN_GROUP_SIZE    = 5     # Минимум наблюдений для статистических критериев

# Название колонки из условия CategoryDelivery, но на предложенных данных колонка называется CategoryNameDelivery.
# Автодетект ниже: если не найдено — подбирается из доступных.
_CAT_DELIVERY_CANDIDATES = [
    "CategoryDelivery",
    "CategoryNameDelivery",
    "category_delivery",
    "categorydelivery",
]


# ═══════════════════════════════════════════════════════════════
# УТИЛИТЫ
# ═══════════════════════════════════════════════════════════════

def _detect_cat_col(df: pd.DataFrame) -> str:
    """Автоматически определяет название колонки CategoryDelivery."""
    for name in _CAT_DELIVERY_CANDIDATES:
        if name in df.columns:
            return name
    # Fallback: ищем по подстроке
    for col in df.columns:
        if "delivery" in col.lower() and "brand" not in col.lower():
            return col
    raise KeyError(
        f"Не найдена колонка CategoryDelivery. "
        f"Доступные колонки: {df.columns.tolist()}"
    )


def _kill_decimals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parquet с типом DECIMAL читается как object с decimal.Decimal внутри.
    Конвертируем все такие колонки и известные числовые поля в float64.
    """
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna()
            if len(sample) > 0 and isinstance(sample.iloc[0], decimal.Decimal):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in ["Weight", "week_weight", "month_weight"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def _anomaly_mask(df: pd.DataFrame, anomaly_pairs: set) -> np.ndarray:
    """Быстрая булева маска строк для исключения через merge (не apply)."""
    ap_df = pd.DataFrame(list(anomaly_pairs), columns=["SubjectID", "researchdate"])
    ap_df["_drop"] = True
    merged = df[["SubjectID", "researchdate"]].merge(ap_df, on=["SubjectID", "researchdate"], how="left")
    return merged["_drop"].fillna(False).values.astype(bool)


# ═══════════════════════════════════════════════════════════════
# ЗАГРУЗКА И ПРЕДОБРАБОТКА
# ═══════════════════════════════════════════════════════════════

def load_parquet_files(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"Parquet-файлы не найдены в: {data_dir}")
    print(f"[INFO] Найдено {len(files)} файл(ов):")
    chunks = []
    for f in files:
        print(f"  - {os.path.basename(f)}")
        chunk = pd.read_parquet(f)
        chunk = _kill_decimals(chunk)
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"[INFO] Загружено строк: {len(df):,}")
    return df


def preprocess(df: pd.DataFrame) -> tuple:
    """
    Возвращает (df_filtered, cat_col) — очищенный датафрейм и имя колонки категории.
    """
    df = df.copy()
    df = _kill_decimals(df)

    cat_col = _detect_cat_col(df)
    print(f"[INFO] Используется колонка категории: '{cat_col}'")

    # Обязательные колонки
    required = ["SubjectID", "researchdate", "BrandID", cat_col,
                "Brand", "Weight", "BrandinDelivery"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Отсутствуют колонки: {missing}\nЕсть: {df.columns.tolist()}")

    df["researchdate"] = pd.to_datetime(df["researchdate"]).dt.date
    df["Weight"] = df["Weight"].astype("float64")

    df = df[df["BrandinDelivery"] == 1].copy()
    df = df[df[cat_col].notna() & (df[cat_col].astype(str).str.strip() != "")].copy()
    print(f"[INFO] После фильтра BrandinDelivery=1 + непустой {cat_col}: {len(df):,} строк")
    return df, cat_col


# ═══════════════════════════════════════════════════════════════
# ВЫЧИСЛЕНИЕ daily_ots
# ═══════════════════════════════════════════════════════════════

def compute_daily_ots(df: pd.DataFrame, cat_col: str) -> pd.DataFrame:
    """daily_ots = Weight × count_rows на уровне (SubjectID, date, BrandID, Category)."""
    grp = (
        df.groupby(["SubjectID", "researchdate", "BrandID", cat_col], sort=False)
        .agg(
            count_rows =("SubjectID", "count"),
            Weight     =("Weight",    "first"),
            Brand      =("Brand",     "first"),
        )
        .reset_index()
    )
    grp["Weight"]    = grp["Weight"].astype("float64")
    grp["daily_ots"] = grp["Weight"] * grp["count_rows"].astype("float64")
    return grp


# ═══════════════════════════════════════════════════════════════
# АЛГОРИТМ ОБНАРУЖЕНИЯ АНОМАЛИЙ
# ═══════════════════════════════════════════════════════════════

def _robust_zscore_arr(arr: np.ndarray) -> np.ndarray:
    """Robust Z-Score через numpy — decimal.Decimal не может попасть сюда."""
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad == 0:
        q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        iqr = q3 - q1
        if iqr == 0:
            return np.zeros(len(arr))
        return (arr - med) / (iqr * 0.7413)
    return (arr - med) / (1.4826 * mad)


def detect_anomalies(ots_df: pd.DataFrame, cat_col: str) -> pd.DataFrame:
    """
    Находит аномальные тройки (SubjectID, researchdate, BrandID).

    Три критерия, все направлены только на ВЫСОКИЙ OTS:
      A) Robust Z-Score >= RZSCORE_THRESHOLD
      B) share >= SHARE_THRESHOLD
      C) Tukey upper fence: daily_ots > Q75 + IQR_FACTOR * IQR
    """
    ots_df = ots_df.copy()
    ots_df["daily_ots"] = pd.to_numeric(ots_df["daily_ots"], errors="coerce").astype("float64")
    ots_df["Weight"]    = pd.to_numeric(ots_df["Weight"],    errors="coerce").astype("float64")
    ots_df = ots_df.dropna(subset=["daily_ots"])

    results = []

    for (researchdate, brand_id, cat), group in ots_df.groupby(
        ["researchdate", "BrandID", cat_col], sort=False
    ):
        total_ots = float(group["daily_ots"].sum())
        if total_ots == 0:
            continue

        arr = group["daily_ots"].values.astype("float64")
        n   = len(arr)

        # Предвычисляем статистику группы один раз
        group = group.copy()
        group["share"] = arr / total_ots

        if n >= MIN_GROUP_SIZE:
            rz_arr   = _robust_zscore_arr(arr)
            q75      = float(np.percentile(arr, 75))
            q25      = float(np.percentile(arr, 25))
            iqr      = q75 - q25
            tukey_up = q75 + IQR_FACTOR * iqr
        else:
            rz_arr   = np.full(n, np.nan)
            tukey_up = np.nan

        group["rz"]       = rz_arr
        group["tukey_up"] = tukey_up

        for idx, (_, row) in enumerate(group.iterrows()):
            reasons, score, threshold = [], 0.0, None

            ots_val   = float(row["daily_ots"])
            share_val = float(row["share"])
            rz_val    = float(row["rz"]) if not np.isnan(row["rz"]) else np.nan

            # ── Критерий A: Robust Z-Score ──
            if not np.isnan(rz_val) and rz_val >= RZSCORE_THRESHOLD:
                reasons.append(f"robust_zscore={rz_val:.2f}>={RZSCORE_THRESHOLD}")
                score     = max(score, rz_val)
                threshold = float(RZSCORE_THRESHOLD)

            # ── Критерий B: доля в бренд-дне ──
            if share_val >= SHARE_THRESHOLD:
                reasons.append(f"share={share_val:.3f}>={SHARE_THRESHOLD}")
                score = max(score, share_val * 10)
                if threshold is None:
                    threshold = float(SHARE_THRESHOLD)

            # ── Критерий C: Tukey upper fence ──
            if not np.isnan(tukey_up) and ots_val > tukey_up and tukey_up > 0:
                reasons.append(f"tukey_fence={ots_val:.1f}>Q75+{IQR_FACTOR}*IQR={tukey_up:.1f}")
                score = max(score, (ots_val - tukey_up) / (tukey_up + 1e-9) * 10)
                if threshold is None:
                    threshold = round(tukey_up, 2)

            if reasons:
                results.append({
                    "SubjectID":        row["SubjectID"],
                    "researchdate":     researchdate,
                    "BrandID":          brand_id,
                    "Brand":            row["Brand"],
                    "CategoryDelivery": cat,
                    "daily_ots":        round(ots_val, 4),
                    "score":            round(score, 4),
                    "threshold":        threshold,
                    "reason":           "; ".join(reasons),
                })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# ПЕРЕСЧЁТ OTS ДО / ПОСЛЕ
# ═══════════════════════════════════════════════════════════════

def compute_ots_by_day(df: pd.DataFrame, anomaly_pairs: set) -> pd.DataFrame:
    before = df.groupby("researchdate")["Weight"].sum().reset_index()
    before.columns = ["researchdate", "ots_before"]
    mask = _anomaly_mask(df, anomaly_pairs)
    after = df[~mask].groupby("researchdate")["Weight"].sum().reset_index()
    after.columns = ["researchdate", "ots_after"]
    return before.merge(after, on="researchdate", how="left").fillna(0)


def compute_category_change(df: pd.DataFrame, anomaly_pairs: set, cat_col: str) -> pd.DataFrame:
    before = df.groupby(cat_col)["Weight"].sum().reset_index()
    before.columns = ["CategoryDelivery", "ots_before"]
    mask = _anomaly_mask(df, anomaly_pairs)
    after = df[~mask].groupby(cat_col)["Weight"].sum().reset_index()
    after.columns = ["CategoryDelivery", "ots_after"]
    merged = before.merge(after, on="CategoryDelivery", how="left").fillna(0)
    merged["pct_change"] = (
        (merged["ots_after"] - merged["ots_before"]) / merged["ots_before"] * 100
    )
    return merged


# ═══════════════════════════════════════════════════════════════
# ОБЯЗАТЕЛЬНЫЕ ГРАФИКИ (п. 8.1)
# ═══════════════════════════════════════════════════════════════

def plot_total_ots_before_after(ots_day: pd.DataFrame, output_path: str):
    """График OTS до/после по дням."""
    fig, ax = plt.subplots(figsize=(14, 5))
    dates = [str(d) for d in ots_day["researchdate"]]
    x = list(range(len(dates)))
    avg_b = float(ots_day["ots_before"].mean())
    avg_a = float(ots_day["ots_after"].mean())
    pct   = avg_a / avg_b * 100 if avg_b > 0 else 100
    ax.plot(x, ots_day["ots_before"] / 1000, "r-o", ms=3, label="OTS_beforeFilter",               lw=1.5)
    ax.plot(x, ots_day["ots_after"]  / 1000, "g-o", ms=3, label="OTS_betweenFilterAndReweighing", lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([d[-2:] for d in dates], fontsize=8)
    ax.set_xlabel("Дата"); ax.set_ylabel("OTS (в тыс.)")
    ax.set_title(
        f"Изменение ежедневного OTS (beforeFiltering - betweenFilteringAndReweighing)\n"
        f"'avg_ots_before' = {avg_b/1000:.2f}, "
        f"'avg_ots_after' = {avg_a/1000:.2f}, % = {pct:.2f}"
    )
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()
    print(f"[PLOT] {output_path}")


def plot_category_ots_change(cat_change: pd.DataFrame, output_path: str):
    """Гистограмма изменения OTS по CategoryDelivery в %."""
    cat_change = cat_change.sort_values("CategoryDelivery")
    fig, ax = plt.subplots(figsize=(max(14, len(cat_change) * 0.55), 6))
    bars = ax.bar(range(len(cat_change)), cat_change["pct_change"], color="steelblue")
    ax.set_xticks(range(len(cat_change)))
    ax.set_xticklabels(cat_change["CategoryDelivery"], rotation=90, fontsize=7)
    ax.set_ylabel("%")
    ax.set_title(
        "Гистограмма изменения суммарного OTS по категориям в % "
        "(beforeFiltering - betweenFilteringAndReweighing)"
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    for bar, val in zip(bars, cat_change["pct_change"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                min(bar.get_height(), 0) - 0.05,
                f"{val:.1f}", ha="center", va="top", fontsize=6)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()
    print(f"[PLOT] {output_path}")


def plot_daily_anomaly_count(anomalies_df: pd.DataFrame, output_path: str):
    """Гистограмма числа аномальных респондентов по дням."""
    if anomalies_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Аномалий не найдено", ha="center", va="center", fontsize=14)
        ax.axis("off")
        plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()
        print(f"[PLOT] {output_path} (пусто)")
        return

    counts = (anomalies_df.groupby("researchdate")["SubjectID"]
              .nunique().reset_index())
    counts.columns = ["researchdate", "n"]
    counts = counts.sort_values("researchdate")
    total_u = int(anomalies_df["SubjectID"].nunique())
    total_p = len(anomalies_df)

    fig, ax = plt.subplots(figsize=(14, 5))
    dates = [str(d) for d in counts["researchdate"]]
    x = list(range(len(dates)))
    ax.bar(x, counts["n"], color="steelblue")
    ax.set_xticks(x); ax.set_xticklabels([d[-2:] for d in dates], fontsize=8)
    ax.set_xlabel("Дата"); ax.set_ylabel("Количество outlier'ов")
    ax.set_title(
        f"Гистограмма количества удалённых респондентов. "
        f"Всего удалено {total_p} пар, из них {total_u} уникальных"
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()
    print(f"[PLOT] {output_path}")


# ═══════════════════════════════════════════════════════════════
# АНАЛИТИЧЕСКИЕ ФУНКЦИИ (п. 8.2)
# ═══════════════════════════════════════════════════════════════

def plot_before_after_by_column(df_full: pd.DataFrame, anomaly_pairs: set,
                                col: str, output_path: str = None):
    """
    График «до/после» по любому столбцу.

    Примеры col:
        Характеристики респондента: 'Пол', 'Возраст', 'Регион',
                                    'Федеральный_округ', 'Занятость',
                                    'Доход', 'Количество_детей'
        Характеристики ресурса:     'ResourceName', 'ResourceType',
                                    'Platform', 'UseType'
        Категории:                  'CategoryNameDelivery', 'Category1',
                                    'Category2', 'Category3'
    """
    if col not in df_full.columns:
        print(f"[WARN] Колонка '{col}' не найдена. Есть: {df_full.columns.tolist()}")
        return
    before = df_full.groupby(col)["Weight"].sum().reset_index()
    before.columns = [col, "ots_before"]
    mask = _anomaly_mask(df_full, anomaly_pairs)
    after = df_full[~mask].groupby(col)["Weight"].sum().reset_index()
    after.columns = [col, "ots_after"]
    merged = before.merge(after, on=col, how="left").fillna(0)
    merged["pct_change"] = (merged["ots_after"] - merged["ots_before"]) / merged["ots_before"] * 100

    n = len(merged)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.7), 5))
    x, w = list(range(n)), 0.35
    ax.bar([i - w/2 for i in x], merged["ots_before"] / 1000, w,
           label="До очистки",    color="steelblue")
    ax.bar([i + w/2 for i in x], merged["ots_after"]  / 1000, w,
           label="После очистки", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(merged[col].astype(str), rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("OTS (в тыс.)"); ax.set_title(f"OTS до/после очистки по: {col}")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150); plt.close(); print(f"[PLOT] {output_path}")
    else:
        plt.show()


def get_query_texts(df_full: pd.DataFrame, subject_id, researchdate) -> pd.DataFrame:
    """
    Поисковые запросы аномального респондента за день.

    Пример:
        rows = get_query_texts(df, 1585271561336880000, '2025-05-07')
        print(rows[['QueryText','Brand','CategoryDelivery','ResourceName']].to_string())
    """
    import datetime
    if isinstance(researchdate, str):
        researchdate = datetime.date.fromisoformat(researchdate)
    mask = (df_full["SubjectID"] == subject_id) & (df_full["researchdate"] == researchdate)
    want = ["SubjectID", "researchdate", "QueryText", "Brand",
            "CategoryDelivery", "ResourceName", "ResourceType", "BrandID", "Weight"]
    # Берём только существующие колонки (имя категории может отличаться)
    avail = [c for c in want if c in df_full.columns]
    # Подставляем реальное имя колонки категории, если нужно
    for cand in _CAT_DELIVERY_CANDIDATES:
        if cand in df_full.columns and cand not in avail:
            avail.append(cand)
    return df_full[mask][avail].reset_index(drop=True)


def plot_brand_ots_before_after(df_full: pd.DataFrame, anomaly_pairs: set,
                                brand_id=None, brand_name: str = None,
                                output_path: str = None):
    """
    OTS конкретного бренда до/после по дням.

    Пример:
        plot_brand_ots_before_after(df, anomaly_pairs, brand_name='redmi',
                                    output_path='output/plots/brand_redmi.png')
    """
    tmp = df_full.copy()
    if brand_id is not None:
        tmp = tmp[tmp["BrandID"].astype(str) == str(brand_id)]
    elif brand_name is not None:
        tmp = tmp[tmp["Brand"].str.lower() == brand_name.lower()]
    else:
        raise ValueError("Укажите brand_id или brand_name")
    if tmp.empty:
        print(f"[WARN] Бренд '{brand_name or brand_id}' не найден."); return

    before = tmp.groupby("researchdate")["Weight"].sum().reset_index()
    before.columns = ["researchdate", "ots_before"]
    mask = _anomaly_mask(tmp, anomaly_pairs)
    after = tmp[~mask].groupby("researchdate")["Weight"].sum().reset_index()
    after.columns = ["researchdate", "ots_after"]
    merged = before.merge(after, on="researchdate", how="left").fillna(0)

    fig, ax = plt.subplots(figsize=(12, 4))
    dates = [str(d) for d in merged["researchdate"]]
    x = list(range(len(dates)))
    ax.plot(x, merged["ots_before"] / 1000, "r-o", ms=4, label="До очистки")
    ax.plot(x, merged["ots_after"]  / 1000, "g-o", ms=4, label="После очистки")
    ax.set_xticks(x); ax.set_xticklabels([d[-2:] for d in dates], fontsize=8)
    ax.set_xlabel("Дата"); ax.set_ylabel("OTS (в тыс.)")
    ax.set_title(f"OTS по бренду '{brand_name or brand_id}' до и после очистки")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150); plt.close(); print(f"[PLOT] {output_path}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════
# МЕТРИКИ ДИАГНОСТИКИ (п. 9)
# ═══════════════════════════════════════════════════════════════

def print_metrics(df: pd.DataFrame, anomalies_df: pd.DataFrame,
                  anomaly_pairs: set, cat_col: str):
    total_subj   = int(df["SubjectID"].nunique())
    anomaly_subj = int(anomalies_df["SubjectID"].nunique()) if not anomalies_df.empty else 0
    ots_before   = float(df["Weight"].sum())
    mask         = _anomaly_mask(df, anomaly_pairs)
    ots_after    = float(df[~mask]["Weight"].sum())
    retention    = ots_after / ots_before * 100 if ots_before > 0 else 100

    avg_per_day = (
        anomalies_df.groupby("researchdate")["SubjectID"].nunique().mean()
        if not anomalies_df.empty else 0
    )

    # Средняя доля аномалий внутри CategoryDelivery
    if not anomalies_df.empty and cat_col in df.columns:
        cat_subj = df.groupby(cat_col)["SubjectID"].nunique().reset_index()
        cat_subj.columns = [cat_col, "total"]
        # аномальные субъекты с категорией из reasons
        anom_cat = (anomalies_df.rename(columns={"CategoryDelivery": cat_col})
                    if "CategoryDelivery" in anomalies_df.columns else anomalies_df)
        cat_subj_anom = (
            anom_cat.groupby(cat_col)["SubjectID"].nunique().reset_index()
            if cat_col in anom_cat.columns
            else pd.DataFrame(columns=[cat_col, "SubjectID"])
        )
        cat_subj_anom.columns = [cat_col, "anomaly"]
        merged = cat_subj.merge(cat_subj_anom, on=cat_col, how="left").fillna(0)
        merged["share"] = merged["anomaly"] / merged["total"].replace(0, np.nan)
        avg_cat_share = float(merged["share"].mean()) * 100
    else:
        avg_cat_share = 0.0

    print("\n" + "═"*55)
    print("  МЕТРИКИ КАЧЕСТВА (п. 9 условия)")
    print("═"*55)
    print(f"  Всего уникальных респондентов:    {total_subj:,}")
    print(f"  Аномальных респондентов:          {anomaly_subj:,} ({anomaly_subj/total_subj*100:.3f}%)")
    print(f"  Уникальных пар SubjectID-date:    {len(anomalies_df):,}")
    print(f"  OTS до очистки:                  {ots_before/1000:,.1f} тыс.")
    print(f"  OTS после очистки:               {ots_after/1000:,.1f} тыс.")
    print(f"  Доля сохранённого OTS:           {retention:.3f}%")
    print(f"  Среднее аномалий в день:         {avg_per_day:.2f}")
    print(f"  Средняя доля аном. по категор.:  {avg_cat_share:.3f}%")
    print("═"*55 + "\n")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Поиск аномальных респондентов в SoS")
    parser.add_argument("--data_dir",   default=".", help="Директория с parquet-файлами")
    parser.add_argument("--output_dir", default="output", help="Директория для результатов")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    # 1. Загрузка
    print("\n[1/6] Загрузка данных...")
    df_raw = load_parquet_files(args.data_dir)

    # 2. Предобработка
    print("[2/6] Предобработка...")
    df, cat_col = preprocess(df_raw)

    # 3. Агрегация daily_ots
    print("[3/6] Вычисление daily_ots...")
    ots_df = compute_daily_ots(df, cat_col)
    print(f"[INFO] Уникальных групп (SubjectID, date, BrandID): {len(ots_df):,}")

    # 4. Обнаружение аномалий
    print("[4/6] Обнаружение аномалий...")
    reasons_df = detect_anomalies(ots_df, cat_col)
    print(f"[INFO] Найдено триггеров аномалий: {len(reasons_df):,}")

    # Уникальные пары SubjectID-date
    if not reasons_df.empty:
        anomalies_df = (reasons_df[["SubjectID", "researchdate"]]
                        .drop_duplicates().reset_index(drop=True))
    else:
        anomalies_df = pd.DataFrame(columns=["SubjectID", "researchdate"])
    print(f"[INFO] Уникальных пар для удаления: {len(anomalies_df):,}")

    anomaly_pairs = set(zip(anomalies_df["SubjectID"], anomalies_df["researchdate"]))

    # 5. Сохранение
    print("[5/6] Сохранение файлов...")
    anomalies_df.to_csv(os.path.join(args.output_dir, "anomalies.csv"), index=False, encoding='utf-8-sig')
    print(f"[SAVE] {args.output_dir}/anomalies.csv")

    if not reasons_df.empty:
        reasons_df.to_csv(os.path.join(args.output_dir, "anomaly_reasons.csv"), index=False, encoding='utf-8-sig')
        print(f"[SAVE] {args.output_dir}/anomaly_reasons.csv")

    # 6. Графики
    print("[6/6] Построение графиков...")
    ots_day = compute_ots_by_day(df, anomaly_pairs)
    plot_total_ots_before_after(
        ots_day,
        os.path.join(args.output_dir, "plots", "total_ots_before_after.png")
    )
    cat_change = compute_category_change(df, anomaly_pairs, cat_col)
    plot_category_ots_change(
        cat_change,
        os.path.join(args.output_dir, "plots", "category_ots_change.png")
    )
    plot_daily_anomaly_count(
        anomalies_df,
        os.path.join(args.output_dir, "plots", "daily_anomaly_count.png")
    )

    # Метрики
    print_metrics(df, anomalies_df, anomaly_pairs, cat_col)
    print("[DONE] Готово! Результаты:", args.output_dir)


if __name__ == "__main__":
    main()
