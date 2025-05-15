import os, joblib, optuna, pandas as pd, numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from datetime import timedelta


CSV_PATH        = "dataset_clean.csv"
RANDOM_STATE    = 42
NA_TOL          = 0.8
LOOKBACK_MONTHS = 6     
D_LAG           = 2
TOP_N_FEATURES  = 6
LAG_RANGE       = [2, 3]  
MIN_ROWS        = 50


df_all = pd.read_csv(CSV_PATH)
df_all["DateTime"] = pd.to_datetime(df_all["DateTime"])
df_all["Hour"] = pd.to_numeric(df_all["Hour"], errors="coerce").astype(int)
df_all["TransactionType"] = df_all["TransactionType"].str.upper().str.strip()

df_all["spread"] = np.where(
    df_all["TransactionType"] == "BID",
    df_all["RTMSPP"] - df_all["DAMSPP"],
    df_all["DAMSPP"] - df_all["RTMSPP"]
)

df_all.sort_values(["AssetID", "Hour", "DateTime"], inplace=True)


targets = []
for (asset, hr, tx), g in df_all.groupby(["AssetID", "Hour", "TransactionType"]):
    if len(g) < MIN_ROWS:
        continue
    precio_mw = g["RTMSPP"] if tx == "BID" else g["DAMSPP"]
    precio_mw_mean = precio_mw.replace(0, np.nan).mean()
    if precio_mw_mean < 5.0 or np.isnan(precio_mw_mean):
        continue
    targets.append((asset, hr, tx))

print(f"Se entrenarán {len(targets)} modelos:", targets)


def compute_lagged_spread_correlations(df_filtered, asset, hour, lag_range, top_n):
    spread_pivot = df_filtered.pivot_table(index="DateTime", columns=["AssetID", "Hour"], values="spread").sort_index()
    target_col = (asset, hour)
    if target_col not in spread_pivot.columns:
        raise ValueError(f"Target ({asset}, {hour}) not found.")
    spread_target = spread_pivot[target_col]
    all_results = []
    for lag in lag_range:
        spread_lagged = spread_pivot.shift(lag)
        valid_cols = spread_lagged.columns[spread_lagged.std() > 0]
        spread_lagged = spread_lagged[valid_cols]
        correlations = spread_lagged.corrwith(spread_target).sort_values(ascending=False).dropna()
        top_corrs = correlations.head(top_n).reset_index()
        top_corrs["Lag"] = lag
        all_results.append(top_corrs)
    final_df = pd.concat(all_results, ignore_index=True)
    return final_df[["Lag", "AssetID", "Hour"]].itertuples(index=False, name=None)

def make_features(df_hist, pivot_hist, external_features, valid_mask=None):
    df = df_hist.copy()
    for lag, node, h in external_features:
        col = f"spread_lag{lag}_{node}_h{h}"
        if (node, h) in pivot_hist.columns:
            df[col] = pivot_hist[(node, h)].shift(lag)
    df["spread_std_3"] = df["spread"].rolling(3).std().shift(2)
    df["spread_std_7"] = df["spread"].rolling(7).std().shift(2)
    df["spread_absdiff_lag2"] = (df["spread"].shift(2) - df["spread"].shift(3)).abs()
    df["shock_flag"] = (df["spread_absdiff_lag2"] > 10).astype(int)
    df["spread_ma_7"] = df["spread"].rolling(7).mean().shift(2)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if valid_mask is None:
        valid_mask = df.notna().mean() >= NA_TOL
    df = df.loc[:, valid_mask].select_dtypes(include="number")
    if "DateTime" in df_hist:
        df["DateTime"] = df_hist["DateTime"]
    if "spread" in df_hist:
        df["spread"] = df_hist["spread"]
    return df, valid_mask


pivot_all = df_all.pivot_table(index="DateTime", columns=["AssetID", "Hour"], values="spread").sort_index()

for asset_id, hour, tx_type in targets:

    df_filtered = df_all[df_all["TransactionType"] == tx_type]
    raw = df_filtered[(df_filtered["AssetID"] == asset_id) & (df_filtered["Hour"] == hour)].copy()
    if len(raw) < MIN_ROWS:
        continue

    raw.reset_index(drop=True, inplace=True)
    first_date = raw["DateTime"].min().normalize()
    train_cut = first_date + pd.DateOffset(months=LOOKBACK_MONTHS)
    cutoff_date = train_cut


    df_corr = df_all[df_all["DateTime"] < cutoff_date][["DateTime", "AssetID", "Hour", "spread"]].dropna()
    external_features = compute_lagged_spread_correlations(df_corr, asset_id, hour, LAG_RANGE, TOP_N_FEATURES)


    hist0 = raw[raw["DateTime"] <= train_cut - timedelta(days=D_LAG)]
    pivot0 = pivot_all.loc[:train_cut - timedelta(days=D_LAG)]
    df0, valid_mask = make_features(hist0, pivot0, external_features)
    df0.dropna(inplace=True)
    FEATURES = [col for col in df0.columns if col not in ["DateTime", "spread"]]
    X0, y0 = df0[FEATURES], df0["spread"]


    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", .005, .05, log=True),
            "subsample": trial.suggest_float("subsample", .5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", .3, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
        }
        tscv = TimeSeriesSplit(n_splits=5)
        return np.mean([
            np.sqrt(mean_squared_error(y0.iloc[val], XGBRegressor(**params).fit(X0.iloc[train], y0.iloc[train]).predict(X0.iloc[val])))
            for train, val in tscv.split(X0)
        ])

    study = optuna.create_study(direction="minimize")
    best_value = float("inf")
    no_improve_counter = 0
    n_trials = 5
    n_trials_no_improve = 3
    for i in range(n_trials):
        study.optimize(objective, n_trials=1, show_progress_bar=False)
        current_value = study.best_value
        if current_value < best_value * 0.995:
            best_value = current_value
            no_improve_counter = 0
        else:
            no_improve_counter += 1
        print(f"Trial {i} | Best RMSE: {best_value:.4f} | No Improvement: {no_improve_counter}")
        if no_improve_counter >= n_trials_no_improve:
            print("🛑 Early stopping triggered.")
            break

  
    model_dir = f"modelos_guardados/{asset_id}_h{hour}_{tx_type}"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(study.best_params,     f"{model_dir}/best_params.joblib")
    joblib.dump(FEATURES,              f"{model_dir}/features.joblib")
    joblib.dump(valid_mask,            f"{model_dir}/valid_mask.joblib")
    joblib.dump(list(external_features), f"{model_dir}/external_features.joblib")
    raw.to_csv(f"{model_dir}/raw.csv", index=False)
    print(f"----------------- Guardado modelo {asset_id} h{hour} {tx_type} -----------------")
