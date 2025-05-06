
import os, joblib, optuna, pandas as pd, numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from datetime import timedelta
from utils_features import compute_lagged_spread_correlations, make_features

class EarlyStoppingCallback:
    def __init__(self, patience=3, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float("inf")
        self.counter = 0

    def __call__(self, study, trial):
        current_best = study.best_value
        if current_best < self.best_value - self.min_delta:
            self.best_value = current_best
            self.counter = 0
            print(f" Trial {trial.number}: mejora detectada → nuevo mejor valor: {current_best:.6f}")
        else:
            self.counter += 1
            print(f" Trial {trial.number}: sin mejora. ({self.counter}/{self.patience})")
        if self.counter >= self.patience:
            print(f"\n🛑 Early stopping activado: no hubo mejora en {self.patience} intentos consecutivos.\n")
            study.stop()


CSV_PATH        = "dataset_clean.csv"
RANDOM_STATE    = 42
LOOKBACK_MONTHS = 6
D_LAG           = 2
TOP_N_FEATURES  = 6
LAG_RANGE       = range(2, 4)
NA_TOL          = 0.8
MIN_ROWS        = 50


df_all = pd.read_csv(CSV_PATH)
df_all["DateTime"] = pd.to_datetime(df_all["DateTime"])
df_all["Hour"]     = pd.to_numeric(df_all["Hour"], errors="coerce").astype(int)
df_all["TransactionType"] = df_all["TransactionType"].str.upper().str.strip()
df_all["NodeType"] = df_all["AssetID"] + "_" + df_all["TransactionType"]
df_all["DateTime_hr"] = df_all["DateTime"].dt.normalize() + pd.to_timedelta(df_all["Hour"], "h")
df_all = df_all[df_all["TransactionType"].isin(["BID", "OFFER"])]


df_all["spread"] = np.where(
    df_all["TransactionType"] == "BID",
    df_all["RTMSPP"] - df_all["DAMSPP"],
    df_all["DAMSPP"] - df_all["RTMSPP"]
)


targets = []
for (asset, hr, tx), g in df_all.groupby(["AssetID", "Hour", "TransactionType"]):
    if len(g) < MIN_ROWS:
        continue

    precio_mw = g["RTMSPP"] if tx == "BID" else g["DAMSPP"]
    precio_mw_mean = precio_mw.replace(0, np.nan).mean()

    if precio_mw_mean < 5.0 or np.isnan(precio_mw_mean):
        print(f" Nodo {asset} h{hr} {tx} excluido: precio promedio de compra {precio_mw_mean:.2f} es muy bajo → MW irreal.")
        continue

    targets.append((asset, hr, tx))

print(f" Se entrenarán {len(targets)} modelos inicialmente:", targets)


MIN_REQUIRED_NODES = 5

if len(targets) < MIN_REQUIRED_NODES:
    print(f" Solo se seleccionaron {len(targets)} nodos. Buscando nodos adicionales para cumplir mínimo de {MIN_REQUIRED_NODES}...")

    all_combos = df_all.groupby(["AssetID", "Hour", "TransactionType"]).size().reset_index(name="count")
    selected_set = set(targets)
    remaining = [tuple(x) for x in all_combos[["AssetID", "Hour", "TransactionType"]].values if tuple(x) not in selected_set]

    remaining.sort(key=lambda x: -all_combos[
        (all_combos["AssetID"] == x[0]) & 
        (all_combos["Hour"] == x[1]) & 
        (all_combos["TransactionType"] == x[2])
    ]["count"].values[0])

    for candidate in remaining:
        if len(targets) >= MIN_REQUIRED_NODES:
            break
        targets.append(candidate)
        print(f"➕ Nodo agregado por cobertura mínima: {candidate}")

print(f" Total de modelos a entrenar tras asegurar mínimo: {len(targets)}")



pivot_all = (
    df_all.pivot_table(index="DateTime_hr", columns=["NodeType", "Hour"], values="spread")
          .sort_index()
)



for asset_id, hour, tx_type in targets:
    model_dir = f"modelos_guardados/{asset_id}_h{hour}_{tx_type}"
    
    if os.path.exists(model_dir):
        print(f" Modelo ya existe: {asset_id} h{hour} {tx_type} — se omite.")
        continue

    raw = df_all[
        (df_all["AssetID"] == asset_id) &
        (df_all["Hour"] == hour) &
        (df_all["TransactionType"] == tx_type)
    ].copy()
    
    if len(raw) < MIN_ROWS:
        continue

    
    first_date = raw["DateTime"].min().normalize()
    train_cut  = first_date + pd.DateOffset(months=LOOKBACK_MONTHS)
    hist_cut   = train_cut - timedelta(hours=1)

    
    hist0  = raw[raw["DateTime_hr"] <= hist_cut]
    pivot0 = pivot_all.loc[:hist_cut]



    
    df_corr = df_all[df_all["DateTime_hr"] < train_cut][
        ["DateTime", "AssetID", "Hour", "TransactionType", "spread"]
    ].dropna()

    external = compute_lagged_spread_correlations(
        df_corr, asset_id, hour,
        LAG_RANGE, TOP_N_FEATURES,
        transaction_type=tx_type
    )

    
    hist0  = raw[raw["DateTime_hr"] <= hist_cut]
    pivot0 = pivot_all.loc[:hist_cut]

    df0, valid_mask = make_features(hist0, pivot0, external, na_tol=NA_TOL)
    
    df0.dropna(inplace=True) 
    
    print(f"\n {asset_id} h{hour} {tx_type} — shape raw: {raw.shape}, pivot0: {pivot0.shape}, external: {len(external)} features")

    
    na_cols = df0.columns[df0.isna().any()].tolist()
    if na_cols:
        print(f" Columnas con NaN en df0 para {asset_id} h{hour} {tx_type}:")
        for col in na_cols:
            count = df0[col].isna().sum()
            print(f"  → {col}: {count} NaNs ({count / len(df0):.2%})")

    
    print(f" valid_mask: {valid_mask.sum()} filas válidas / {len(valid_mask)} totales")
    
    
    feat_cols = [c for c in df0.columns if c not in ("DateTime", "spread")]
    X0, y0    = df0[feat_cols], df0["spread"]

    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", .005, .05, log=True),
            "subsample": trial.suggest_float("subsample", .6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", .3, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": RANDOM_STATE
        }

        cv = TimeSeriesSplit(n_splits=5)
        rmse = []

        for tr, val in cv.split(X0):
            X_train, y_train = X0.iloc[tr], y0.iloc[tr]

            
            invalid_y = y_train[y_train.isna() | np.isinf(y_train) | (np.abs(y_train) > 1e10)]
            if not invalid_y.empty:
                print(f"\n Invalid labels detected in y_train (trial {trial.number}):")
                print(invalid_y)
                print("Corresponding X_train rows:")
                print(X_train.loc[invalid_y.index])
                raise ValueError("Aborting trial due to invalid labels in y_train.")

            invalid_X = X_train[X_train.isna().any(axis=1) | np.isinf(X_train).any(axis=1)]
            if not invalid_X.empty:
                print(f"\n Invalid rows detected in X_train (trial {trial.number}):")
                print(invalid_X)
                raise ValueError("Aborting trial due to invalid features in X_train.")

            
            model = XGBRegressor(**params)
            
            
            if X_train.isna().any().any():
                cols_with_nan = X_train.columns[X_train.isna().any()].tolist()
                print(f" Eliminando columnas con NaNs en X_train: {cols_with_nan}")
                X_train = X_train.drop(columns=cols_with_nan)
                X_val = X0.iloc[val].drop(columns=cols_with_nan)
            else:
                X_val = X0.iloc[val]
            
            
            
            model.fit(X_train, y_train)
            pred = model.predict(X0.iloc[val])
            rmse.append(np.sqrt(mean_squared_error(y0.iloc[val], pred)))

        return np.mean(rmse)

    
    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(
            objective,
            n_trials=20,
            show_progress_bar=False,
            callbacks=[EarlyStoppingCallback(patience=3)]
        )
    except ValueError as e:
        print(f" Nodo {asset_id} h{hour} {tx_type} omitido por error en los datos: {e}")
        continue
    

    
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(study.best_params,     f"{model_dir}/best_params.joblib")
    joblib.dump(feat_cols,             f"{model_dir}/features.joblib")
    joblib.dump(valid_mask,            f"{model_dir}/valid_mask.joblib")
    joblib.dump(external,              f"{model_dir}/external_features.joblib")

    raw["DateTime_hr"] = raw["DateTime"].dt.normalize() + pd.to_timedelta(raw["Hour"], unit="h")
    raw.to_csv(f"{model_dir}/raw.csv", index=False)

    print(f"----------------- Guardado modelo {asset_id} h{hour} {tx_type}-----------------")
