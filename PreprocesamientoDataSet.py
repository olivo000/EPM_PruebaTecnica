import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict


PRICE_MIN, PRICE_MAX = -250, 500

PRICE_COLS           = ["BidPrice", "OfferPrice", "DAMSPP", "RTMSPP"]
MEASURE_COLS         = ["Price", "MW", "DAMSPP", "RTMSPP"]

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

SPREAD_THRESHOLD = 500  


missing_dates_counts = defaultdict(int)


def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype=str)

def parse_datetime(df: pd.DataFrame) -> None:
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")

def normalize_text(df: pd.DataFrame) -> None:
    df["AssetID"] = df["AssetID"].astype(str).str.strip()
    df["TransactionType"] = (
        df["TransactionType"]
        .astype(str).str.strip().str.upper()
        .replace({"OFER": "OFFER", "BID ": "BID", "OFFER ": "OFFER"})
    )

def clean_irrelevant_fields(df):
    is_bid = df["TransactionType"].str.upper().str.strip() == "BID"
    is_offer = df["TransactionType"].str.upper().str.strip() == "OFFER"
    df.replace("", np.nan, inplace=True)
    df.loc[is_bid, ["OfferPrice", "MWOffer"]] = np.nan
    df.loc[is_offer, ["BidPrice", "MWBid"]] = np.nan

def diagnose_irrelevant_columns(df):
    is_bid = df["TransactionType"].str.upper().str.strip() == "BID"
    is_offer = df["TransactionType"].str.upper().str.strip() == "OFFER"
    problems = []
    for col in ["OfferPrice", "MWOffer"]:
        count = df.loc[is_bid, col].notna().sum()
        if count > 0:
            problems.append((col, "BID", count))
    for col in ["BidPrice", "MWBid"]:
        count = df.loc[is_offer, col].notna().sum()
        if count > 0:
            problems.append((col, "OFFER", count))
    if problems:
        print("\n Diagnóstico de columnas no esperadas por tipo de transacción:")
        for col, tx_type, count in problems:
            print(f"  {count} valores encontrados en '{col}' para transacciones tipo {tx_type}")
    else:
        print("\n No se encontraron columnas con valores no esperados según el tipo de transacción.")

def fix_bid_offer_columns(df: pd.DataFrame) -> None:
    is_bid   = df["TransactionType"] == "BID"
    is_offer = df["TransactionType"] == "OFFER"
    df.loc[is_bid & df["BidPrice"].isna(), "BidPrice"] = df.loc[is_bid, "OfferPrice"]
    df.loc[is_offer & df["OfferPrice"].isna(), "OfferPrice"] = df.loc[is_offer, "BidPrice"]
    df["Price"] = df["BidPrice"].fillna(df["OfferPrice"])
    df["MW"]    = df["MWBid"].fillna(df["MWOffer"])

def clip_prices(df: pd.DataFrame) -> None:
    for col in PRICE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(PRICE_MIN, PRICE_MAX)


def validate_mws(df: pd.DataFrame) -> None:
    for col in ["MWBid", "MWOffer"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round()
            df.loc[df[col] < 0, col] = np.nan



def mark_extreme_spreads(df: pd.DataFrame) -> None:
    is_bid   = df["TransactionType"] == "BID"
    is_offer = df["TransactionType"] == "OFFER"


    spread = pd.Series(index=df.index, dtype=float)
    spread[is_bid]   = pd.to_numeric(df.loc[is_bid,   "RTMSPP"], errors="coerce") \
                     - pd.to_numeric(df.loc[is_bid,   "DAMSPP"], errors="coerce")
    spread[is_offer] = pd.to_numeric(df.loc[is_offer, "DAMSPP"], errors="coerce") \
                     - pd.to_numeric(df.loc[is_offer, "RTMSPP"], errors="coerce")

    df["spread"] = spread


    df["ExtremeSpreadFlag"] = df["spread"].abs() > SPREAD_THRESHOLD
    df["ExtremeSpreadFlag"] = df["ExtremeSpreadFlag"].fillna(False).astype(bool)


    df.loc[df["ExtremeSpreadFlag"], ["DAMSPP", "RTMSPP"]] = 0
    df.loc[df["ExtremeSpreadFlag"],  "spread"]            = 0.0


    for col in ["DAMSPP", "RTMSPP", "MW"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")


    
    mask_fill = pd.Series([False] * len(df), index=df.index)
    
    if "RowAdded" in df.columns:
        mask_fill = df["RowAdded"] & df["DAMSPP"].isna() & df["RTMSPP"].isna()
        df.loc[mask_fill, "DAMSPP"] = 20.0
        df.loc[mask_fill, "RTMSPP"] = 20.001


    df["spread"] = np.where(
        df["TransactionType"] == "BID",
        df["RTMSPP"] - df["DAMSPP"],
        df["DAMSPP"] - df["RTMSPP"]
    )


    df["ReturnUSD"] = (df["RTMSPP"] - df["DAMSPP"]) * df["MW"].fillna(0)

    print(
        f"  Se marcaron {df['ExtremeSpreadFlag'].sum()} filas con spread "
        f"extremo (> {SPREAD_THRESHOLD})"
    )
    print(
        f" Se rellenaron {mask_fill.sum()} filas añadidas con RTMSPP ≈ DAMSPP + ε"
    )





def ensure_continuous_time_index(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = df["DateTime"].dt.floor("D")
    df["Hour"] = pd.to_numeric(df["DateTime"].dt.hour, errors="coerce")

    out_frames = []
    for (asset, hour), g in df.groupby(["AssetID", "Hour"]):
        try:
            hour_int = int(hour)
            if not 0 <= hour_int <= 23:
                continue
        except (ValueError, TypeError):
            continue

        idx = pd.date_range(g["Date"].min(), g["Date"].max(), freq="D")
        template = pd.DataFrame({"Date": idx, "Hour": hour_int, "AssetID": asset})
        template["DateTime"] = template["Date"] + pd.to_timedelta(hour_int, unit="h")

        merged = template.merge(
            g, on=["AssetID", "Hour", "Date", "DateTime"], how="left", indicator=True
        )
        merged["RowAdded"] = merged["_merge"].eq("left_only")
        merged.drop(columns="_merge", inplace=True)

        # NUEVO: guardar la fecha de fila añadida
        if "RowAddedDate" not in merged.columns:
            merged["RowAddedDate"] = pd.NaT
        merged.loc[merged["RowAdded"], "RowAddedDate"] = merged.loc[merged["RowAdded"], "Date"]

        # contar días vacíos
        missing_all = (
            merged.loc[merged["RowAdded"], MEASURE_COLS].isna().all(axis=1).sum()
        )
        if missing_all:
            missing_dates_counts[(asset, hour_int)] += missing_all

        out_frames.append(merged)

    return pd.concat(out_frames, ignore_index=True)

def rolling_mean_sequential(df, cols, window=30, min_gap=2):
    for col in cols:
        for (asset, hr), g in df.groupby(["AssetID", "Hour"], sort=False):
            idx = g.index
            series = g[col].copy()
            hist_l = []

            for i in range(len(series)):
                if pd.isna(series.iat[i]):
                    past = series.iloc[max(0, i-window-min_gap): i-min_gap].dropna()
                    hist_l.append(len(past))
                    if len(past):
                        series.iat[i] = past.mean()
                else:
                    hist_l.append(0)

            df.loc[idx, col] = series
            df.loc[idx, f"{col}_ValidHistoryLen"] = hist_l

def recompute_returnusd(df: pd.DataFrame) -> None:
    df["DAMSPP"]  = pd.to_numeric(df["DAMSPP"], errors="coerce")
    df["RTMSPP"]  = pd.to_numeric(df["RTMSPP"], errors="coerce")
    df["MW"]      = pd.to_numeric(df["MW"], errors="coerce")
    df["ReturnUSD"] = (df["RTMSPP"] - df["DAMSPP"]) * df["MW"].fillna(0)

def mark_invalid_rows(df: pd.DataFrame) -> None:
    df["MissingFlag"] = (
        df["DateTime"].isna() |
        df["AssetID"].isna()  |
        ~df["TransactionType"].isin(["BID", "OFFER"]) |
        df["Price"].isna()    |
        df["DAMSPP"].isna()   |
        df["RTMSPP"].isna()
    )

def check_missing_summary(df: pd.DataFrame) -> None:
    print("\n🔍 Valores faltantes tras imputación:")

    # Revisión individual de columnas
    for col in MEASURE_COLS:
        missing_count = df[col].isna().sum()
        print(f"- {col}: {missing_count} NaNs")

    # Imputación especial: si DAMSPP o RTMSPP es NaN y el otro no lo es
    bid_mask = df["TransactionType"] == "BID"
    offer_mask = df["TransactionType"] == "OFFER"

    # BID: DAMSPP está bien, RTMSPP falta → RTMSPP = DAMSPP + 0.1
    bid_fix_mask = bid_mask & df["RTMSPP"].isna() & df["DAMSPP"].notna()
    df.loc[bid_fix_mask, "RTMSPP"] = df.loc[bid_fix_mask, "DAMSPP"] + 0.1

    # BID: RTMSPP está bien, DAMSPP falta → DAMSPP = RTMSPP - 0.1
    bid_fix_mask_2 = bid_mask & df["DAMSPP"].isna() & df["RTMSPP"].notna()
    df.loc[bid_fix_mask_2, "DAMSPP"] = df.loc[bid_fix_mask_2, "RTMSPP"] - 0.1

    # OFFER: RTMSPP está bien, DAMSPP falta → DAMSPP = RTMSPP + 0.1
    offer_fix_mask = offer_mask & df["DAMSPP"].isna() & df["RTMSPP"].notna()
    df.loc[offer_fix_mask, "DAMSPP"] = df.loc[offer_fix_mask, "RTMSPP"] + 0.1

    # OFFER: DAMSPP está bien, RTMSPP falta → RTMSPP = DAMSPP - 0.1
    offer_fix_mask_2 = offer_mask & df["RTMSPP"].isna() & df["DAMSPP"].notna()
    df.loc[offer_fix_mask_2, "RTMSPP"] = df.loc[offer_fix_mask_2, "DAMSPP"] - 0.1

    # Recalcula el spread después de corregir
    df["spread"] = np.where(
        df["TransactionType"] == "BID",
        df["RTMSPP"] - df["DAMSPP"],
        df["DAMSPP"] - df["RTMSPP"]
    )

def generate_completeness_report(df: pd.DataFrame, P=30):
    rows = []
    for (asset, hour), g in df.groupby(["AssetID", "Hour"]):
        key = (asset, hour)
        rows.append({
            "AssetID": asset,
            "Hour": hour,
            "TotalDays": len(g),
            "RowsAdded":          int(g["RowAdded"].sum()),
            "RowsAddedAndFilled": int(g["RowAddedAndFilled"].sum()),
            "Missing_DaysAllValues": missing_dates_counts.get(key, 0),
            **{f"Missing_{col}": g[col].isna().sum() for col in MEASURE_COLS},
            **{f"{col}_DaysWith≥{P}": (
                g.get(f"{col}_ValidHistoryLen", pd.Series(dtype=int)) >= P
            ).sum() for col in MEASURE_COLS}
        })

    report = pd.DataFrame(rows)
    now = datetime.now().strftime("%Y%m%d_%H%M")
    report_path = LOG_DIR / f"completeness_report_{now}.csv"
    report.to_csv(report_path, index=False)

    print("\n Reporte de Completitud Generado:")


def export_added_dates(df: pd.DataFrame):
    added = df.loc[df["RowAdded"], ["AssetID", "Hour", "Date", "RowAddedAndFilled"]]
    now = datetime.now().strftime("%Y%m%d_%H%M")
    detail_path = LOG_DIR / f"added_days_detail_{now}.csv"
    added.to_csv(detail_path, index=False)
    print(f"\n🗂️ Detalle guardado en: {detail_path.name}")
def export_integrity_ranking(df: pd.DataFrame):
    """
    Exporta tabla de integridad real por (AssetID, Hour), considerando:
    - Días añadidos (por completitud temporal)
    - Días inválidos (por datos internos incompletos)
    """
    df_copy = df.copy()

    # Asegura tipos booleanos correctos para agrupación
    df_copy["RowAdded"]     = df_copy["RowAdded"].fillna(False).astype(bool)
    df_copy["MissingFlag"]  = df_copy["MissingFlag"].fillna(False).astype(bool)

    grouped = df_copy.groupby(["AssetID", "Hour"]).agg(
        TotalDays        = ("Date", "count"),
        DaysAdded        = ("RowAdded", "sum"),
        InvalidDays      = ("MissingFlag", "sum"),
        DaysFilled       = ("RowAddedAndFilled", "sum"),
    ).reset_index()

    grouped["EffectiveDays"] = grouped["TotalDays"] - grouped["InvalidDays"]
    grouped["FullIntegrityScore"] = grouped["EffectiveDays"] / grouped["TotalDays"]
    grouped.sort_values(by="FullIntegrityScore", ascending=False, inplace=True)

    now = datetime.now().strftime("%Y%m%d_%H%M")
    path = LOG_DIR / f"integrity_ranking_{now}.csv"
    grouped.to_csv(path, index=False)

    print(f"\nTabla de integridad exportada como: {path.name}")
    print(grouped[["AssetID", "Hour", "TotalDays", "DaysAdded", "InvalidDays", "FullIntegrityScore"]].head(10).to_string(index=False))


# ──────────────────────────  MAIN  ────────────────────────────────────────────
def main():
    df = load_data("dataset_pt_20250428v0.csv")
    parse_datetime(df)
    normalize_text(df)
    clean_irrelevant_fields(df)
    diagnose_irrelevant_columns(df)
    fix_bid_offer_columns(df)

    clip_prices(df)
    

    validate_mws(df)
    df = ensure_continuous_time_index(df)
    mark_extreme_spreads(df)

    for col in MEASURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rolling_mean_sequential(df, MEASURE_COLS, window=30, min_gap=2)

    df["RowAddedAndFilled"] = df["RowAdded"] & df[MEASURE_COLS].notna().any(axis=1)

    recompute_returnusd(df)
    df["MW"] = df["MW"].round().astype("Int64")  
    mark_invalid_rows(df)
    check_missing_summary(df)

    df.to_csv("dataset_clean.csv", index=False)
    generate_completeness_report(df)
    export_added_dates(df)
    export_integrity_ranking(df)





    print(
        f"\n  Días añadidos totales: {df['RowAdded'].sum()} "
        f"(rellenados: {df['RowAddedAndFilled'].sum()})"
    )
    print("\nPrimeras filas con columnas clave:")
    print(df[["DateTime", "AssetID", "Hour", "TransactionType", "spread", "ExtremeSpreadFlag", "RowAdded", "RowAddedAndFilled"]].head(10))

    print(df["spread"].describe())


if __name__ == "__main__":
    main()
