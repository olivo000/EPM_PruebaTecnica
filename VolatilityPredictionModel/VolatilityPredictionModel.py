import pandas as pd, numpy as np, optuna, matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import matplotlib.dates as mdates

# ─── Configuración ───
CSV_PATH        = "dataset_BID.csv"
VOL_WINDOW      = 5
RANDOM_STATE    = 42
NA_TOL          = 0.8
LOOKBACK_MONTHS = 7
D_LAG           = 2
TOP_N_FEATURES  = 6
LAG_RANGE       = [2, 3]
TARGETS = [("LSBRIGHTSD_RN-SP", 5)]

# ─── Carga base ───
df_all = pd.read_csv(CSV_PATH)
df_all["DateTime"] = pd.to_datetime(df_all["DateTime"])
df_all["Hour"]     = pd.to_numeric(df_all["Hour"], errors="coerce").astype(int)
df_all             = df_all[df_all["TransactionType"].str.upper().str.strip() == "BID"]

# SPREAD y VOLATILIDAD
df_all["spread"] = df_all["DAMSPP"] - df_all["RTMSPP"]
df_all["volatility"] = (
    df_all.groupby(["AssetID", "Hour"])["spread"]
          .transform(lambda s: s.shift(2).rolling(VOL_WINDOW, min_periods=3).std())
)
df_all["sqrt_volatility"] = np.sqrt(df_all["volatility"])
df_all.sort_values(["AssetID", "Hour", "DateTime"], inplace=True)

def compute_lagged_vol_correlations(df_filtered, asset, hour, lag_range, top_n):
    v_pivot = df_filtered.pivot_table(index="DateTime",
                                      columns=["AssetID", "Hour"],
                                      values="volatility").sort_index()
    tgt = v_pivot[(asset, hour)]
    out = []
    for lag in lag_range:
        v_lag = v_pivot.shift(lag)
        valid_cols = v_lag.std()[v_lag.std() > 0].index
        v_lag = v_lag[valid_cols]
        cors = v_lag.corrwith(tgt).dropna().sort_values(ascending=False)
        top = cors.head(top_n).reset_index()
        top["Lag"] = lag
        out.append(top)
    return pd.concat(out, ignore_index=True)[["Lag", "AssetID", "Hour"]].itertuples(index=False, name=None)


def make_features(df_hist, pivot_hist, ext_feats, valid_mask=None):
    df = df_hist.copy()

    # ─── External lagged features ─────────────────────────────────────────────
    for lag, node, h in ext_feats:
        col = f"vol_lag{lag}_{node}_h{h}"
        if (node, h) in pivot_hist.columns:
            df[col] = np.sqrt(pivot_hist[(node, h)]).shift(lag)

    # Base desplazada
    sqrt_vol = df["sqrt_volatility"]
    sqrt_vol_shift2 = sqrt_vol.shift(2)

    # ─── Estadísticas móviles basadas solo en días ≤ d-2 ─────────────────────
    df["sqrt_vol_std_3"] = sqrt_vol.rolling(3).std().shift(2)
    df["sqrt_vol_std_7"] = sqrt_vol.rolling(7).std().shift(2)
    df["sqrt_vol_ma_7"]  = sqrt_vol.rolling(7).mean().shift(2)

    # ─── Indicadores diferenciales ───────────────────────────────────────────
    df["sqrt_vol_diff_lag2"] = (sqrt_vol.shift(2) - sqrt_vol.shift(3)).abs()
    df["shock_flag"] = (df["sqrt_vol_diff_lag2"] > df["sqrt_vol_std_7"]).astype(int)

    # ─── Transformaciones del logaritmo sin fuga ────────────────────────────
    df["log_sqrt_vol_lag2"] = np.log1p(sqrt_vol_shift2)
    df["log_sqrt_vol_squared"] = df["log_sqrt_vol_lag2"] ** 2
    df["log_sqrt_vol_quad"]    = df["log_sqrt_vol_squared"] ** 2

    # ─── Interacciones y razones ─────────────────────────────────────────────
    df["log_vol_ma7_x_shock"] = df["sqrt_vol_ma_7"] * df["shock_flag"]
    df["sqrt_vol_std_ratio"] = df["sqrt_vol_std_3"] / (df["sqrt_vol_std_7"] + 1e-6)
    df["sqrt_vol_diff_vs_ma7"] = sqrt_vol.shift(2) - df["sqrt_vol_ma_7"]

    # ─── NUEVAS FEATURES PRO (basadas en sqrt_volatility) ───────────────────

    # 1) Polinomio sobre razón de std
    df["sqrt_vol_std_ratio_sq"] = df["sqrt_vol_std_ratio"] ** 2

    # 2) Z-score ventana 14
    mu_14 = sqrt_vol_shift2.rolling(14).mean()
    std_14 = sqrt_vol_shift2.rolling(14).std()
    df["sqrt_vol_zscore_14"] = (sqrt_vol_shift2 - mu_14) / (std_14 + 1e-6)

    # 3) EWMA (span 5)
    df["sqrt_vol_ewm_5_lag2"] = sqrt_vol_shift2.ewm(span=5).mean()

    # 4) Proxy de entropía local
    vol_window = sqrt_vol_shift2.rolling(5)
    df["sqrt_vol_entropy_proxy"] = vol_window.apply(
        lambda x: np.mean(np.abs(np.diff(x))) / (np.std(x) + 1e-6), raw=True
    )

    # 5) Aceleración de la sqrt_vol (segunda derivada discreta)
    df["sqrt_vol_acceleration"] = (
        sqrt_vol_shift2 - 2 * sqrt_vol.shift(3) + sqrt_vol.shift(4)
    )

    # 6) Skewness local (asimetría)
    vol_5 = sqrt_vol_shift2.rolling(5)
    df["sqrt_vol_skew_proxy"] = vol_5.apply(
        lambda x: np.mean((x - np.mean(x)) ** 3) / (np.std(x) ** 3 + 1e-6), raw=True
    )

    # ─── Limpieza final ───────────────────────────────────────────────────────
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if valid_mask is None:
        valid_mask = df.notna().mean() >= NA_TOL

    num_df = df.loc[:, valid_mask].select_dtypes(include="number").copy()
    num_df["DateTime"] = df_hist["DateTime"]

    return num_df, valid_mask


# ─── Entrenamiento ───
models_info = []
pivot_all = df_all.pivot_table(index="DateTime", columns=["AssetID", "Hour"], values="volatility").sort_index()

for asset_id, hour in TARGETS:
    raw = df_all[(df_all["AssetID"] == asset_id) & (df_all["Hour"] == hour)].copy()
    raw.reset_index(drop=True, inplace=True)
    first_date = raw["DateTime"].min().normalize()
    train_cut = first_date + pd.DateOffset(months=LOOKBACK_MONTHS)
    cutoff_date = first_date + pd.DateOffset(months=LOOKBACK_MONTHS)
    df_corr = df_all[df_all["DateTime"] < cutoff_date][["DateTime", "AssetID", "Hour", "volatility"]].dropna()
    external_features = compute_lagged_vol_correlations(df_corr, asset_id, hour, LAG_RANGE, TOP_N_FEATURES)
    hist0 = raw[raw["DateTime"] <= train_cut - timedelta(days=D_LAG)]
    pivot0 = pivot_all.loc[:train_cut - timedelta(days=D_LAG)]
    df0, valid_mask = make_features(hist0, pivot0, external_features)
    FEATURES = [c for c in df0.columns if c not in ["DateTime", "sqrt_volatility"]]
    valid_rows = df0["sqrt_volatility"].replace([np.inf, -np.inf], np.nan).notna()
    X0 = df0.loc[valid_rows, FEATURES]
    y0 = df0.loc[valid_rows, "sqrt_volatility"]

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
    study.optimize(objective, n_trials=10)

    best_params = study.best_params.copy()
    models_info.append({
        "asset_id": asset_id,
        "hour": hour,
        "raw": raw,
        "features": FEATURES,
        "valid_mask": valid_mask,
        "external_features": external_features,
        "best_params": best_params,
        "dates": [],
        "preds": [],
        "actuals": []
    })
    


# ─── Walk-forward ───
test_start = max(m["raw"]["DateTime"].min().normalize() + pd.DateOffset(months=LOOKBACK_MONTHS) for m in models_info)
test_end = min(m["raw"]["DateTime"].max().normalize() for m in models_info)
current_day = test_start

print("\n# ─── WALK-FORWARD sincronizado ───")
while current_day <= test_end:
    for model in models_info:
        raw = model["raw"]
        feat_cols = model["features"]
        external_features = model["external_features"]
        valid_mask = model["valid_mask"]
        best_params = model["best_params"]

        hist_cut = current_day - timedelta(days=D_LAG)
        hist_df = raw[raw["DateTime"] <= hist_cut]
        pivot_cut = pivot_all.loc[:hist_cut]
        feat_hist, _ = make_features(hist_df, pivot_cut, external_features, valid_mask)
        feat_hist = feat_hist.dropna(subset=feat_cols + ["sqrt_volatility"])
        if len(feat_hist) < 50:
            continue

        X_train, y_train = feat_hist[feat_cols], feat_hist["sqrt_volatility"]

        model_xgb = XGBRegressor(**best_params, random_state=RANDOM_STATE)
        model_xgb.fit(X_train, y_train)

        day_row = raw[raw["DateTime"].dt.normalize() == current_day]
        if day_row.empty:
            continue

        aux_df = pd.concat([hist_df, day_row])
        feat_day, _ = make_features(aux_df, pivot_all.loc[:current_day - timedelta(days=2)], external_features, valid_mask)
        X_day = feat_day.loc[feat_day["DateTime"].dt.normalize() == current_day, feat_cols]
        if X_day.isna().any().any():
            continue

        # 🚩 Mantén predicción y real en sqrt_volatility
        y_pred = model_xgb.predict(X_day)[0]
        y_real = day_row["sqrt_volatility"].values[0]

        model["dates"].append(current_day)
        model["preds"].append(y_pred)
        model["actuals"].append(y_real)
        print(f"▶️ {model['asset_id']}-h{model['hour']} | {current_day.date()} | predicho = {y_pred:.2f}, real = {y_real:.2f} | error = {abs(y_pred - y_real):.2f}")
    current_day += timedelta(days=1)


# ─── Graficar resultados ───
for model in models_info:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # √volatilidad
    ax1.plot(model["dates"], model["actuals"], label="Sqrt Volatilidad Real", linewidth=2)
    ax1.plot(model["dates"], model["preds"], "--", label="Sqrt Volatilidad Predicha")
    ax1.set_ylabel("Sqrt Volatilidad")
    ax1.set_title(f"{model['asset_id']}-h{model['hour']}: Sqrt Volatilidad Real vs Predicha")
    ax1.grid(True)
    ax1.legend()

    # Spread
    asset = model["asset_id"]
    hour = model["hour"]
    df_spread = df_all[(df_all["AssetID"] == asset) & (df_all["Hour"] == hour)][["DateTime", "spread"]].copy()
    df_spread["DateTime"] = df_spread["DateTime"].dt.normalize()
    spread_aligned = pd.DataFrame({"DateTime": pd.to_datetime(model["dates"])})
    spread_aligned = spread_aligned.merge(df_spread, on="DateTime", how="left")

    ax2.plot(spread_aligned["DateTime"], spread_aligned["spread"], color="gray", label="Spread")
    ax2.set_ylabel("Spread")
    ax2.set_title("Comportamiento del Spread")
    ax2.grid(True)
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
