import pandas as pd
import numpy as np

def compute_lagged_spread_correlations(df_corr, target_asset, target_hour,
                                       lag_range, top_n=5,
                                       transaction_type=None,
                                       min_pairs=30):
    df = df_corr.copy()
    df["TransactionType"] = df["TransactionType"].str.upper().str.strip()
    df["NodeType"] = df["AssetID"] + "_" + df["TransactionType"]
    df["DateTime_hr"] = df["DateTime"].dt.normalize() + pd.to_timedelta(df["Hour"], "h")

    pivot = df.pivot_table(index="DateTime_hr",
                           columns=["NodeType", "Hour"],
                           values="spread").sort_index()

    tgt = (f"{target_asset}_{transaction_type.upper()}", target_hour)
    if tgt not in pivot.columns:
        raise ValueError(f"Objetivo {tgt} no encontrado")

    target = pivot[tgt]
    results = []

    for lag in lag_range:
        lagged = pivot.shift(lag)
        for col in lagged.columns:
            aligned = pd.concat([target, lagged[col]], axis=1).dropna()
            if len(aligned) < min_pairs or aligned.iloc[:, 1].std() == 0:
                continue
            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            if pd.notna(corr):
                results.append({"Lag": lag, "NodeType": col[0], "Hour": col[1],
                                "Correlation": corr})

    if not results:
        return []

    df_res = (pd.DataFrame(results)
                .sort_values("Correlation", ascending=False)
                .head(top_n))
    return list(df_res[["Lag", "NodeType", "Hour"]].itertuples(index=False, name=None))


def make_features(df_hist, pivot_hist, external_features,
                  valid_mask=None, na_tol=0.8):

    df = df_hist.copy()

    if "DateTime_hr" not in df:
        df["DateTime_hr"] = df["DateTime"].dt.normalize() + pd.to_timedelta(df["Hour"], "h")
    df.sort_values("DateTime_hr", inplace=True)
    df.set_index("DateTime_hr", inplace=True)



    for lag, node, h in external_features:
        col = f"spread_lag{lag}_{node}_h{h}"
        if (node, h) in pivot_hist.columns:
            df[col] = pivot_hist[(node, h)].shift(lag).reindex(df.index)
        else:
            print(f"  Nodo {node} h{h} no encontrado en pivot → columna {col} será NaN")

    df["spread_std_3"] = df["spread"].rolling(3).std().shift(2)
    df["spread_std_7"] = df["spread"].rolling(7).std().shift(2)
    df["spread_absdiff_lag2"] = (df["spread"].shift(2) - df["spread"].shift(3)).abs()
    df["shock_flag"] = (df["spread_absdiff_lag2"] > 10).astype(int)
    df["spread_ma_7"] = df["spread"].rolling(7).mean().shift(2)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)


    nan_cols = df.columns[df.isna().any()]
    for col in nan_cols:
        count = df[col].isna().sum()


    nan_rows = df[df.isna().any(axis=1)].copy()
    if not nan_rows.empty:
        debug_rows_path = "debug_nan_rows.csv"
        nan_rows.to_csv(debug_rows_path)


    if valid_mask is None:
        valid_mask = df.notna().mean() >= na_tol


    df = df.loc[:, valid_mask].select_dtypes("number")
    df.reset_index(inplace=True)
    df.rename(columns={"DateTime_hr": "DateTime"}, inplace=True)

    return df, valid_mask
