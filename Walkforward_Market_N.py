import os, pandas as pd, numpy as np, joblib, matplotlib.pyplot as plt
from datetime import timedelta
from xgboost import XGBRegressor
import time
from collections import defaultdict

LOOKBACK_MONTHS     = 6
D_LAG               = 2
RANDOM_STATE        = 42
MAX_WERR            = 1.0

CAPITAL_POR_NODO    = 5_000
PESOS_ERR           = np.array([0.2, 0.3, 0.5])

MIN_EXPECTED_PROFIT = 100
MW_MAXIMO           = 5000

CSV_PATH = "dataset_clean.csv"
os.makedirs("logs", exist_ok=True)


CAPITAL_TOTAL_DIA   = 50_000          
MIN_NODOS_DIA       = 5              
TOP_N_SPREADS       = 10            
CAPITAL_POR_NODO    = CAPITAL_TOTAL_DIA // TOP_N_SPREADS  


asignaciones_all = []                 



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

def get_dam_hist(raw, current_day, hour, offset=2):
    fecha_hist = current_day - timedelta(days=offset)
    fila = raw[(raw["DateTime"].dt.normalize() == fecha_hist) & (raw["Hour"] == hour)]
    if not fila.empty:
        return fila["DAMSPP"].values[0]
    else:
        return np.nan



df_all = pd.read_csv(CSV_PATH)
df_all["DateTime"] = pd.to_datetime(df_all["DateTime"])
df_all["Hour"] = pd.to_numeric(df_all["Hour"], errors="coerce").astype(int)
df_all["TransactionType"] = df_all["TransactionType"].str.upper().str.strip()
df_all = df_all[df_all["TransactionType"].isin(["BID", "OFFER"])]


df_all["spread"] = np.where(
    df_all["TransactionType"] == "BID",
    df_all["RTMSPP"] - df_all["DAMSPP"],
    df_all["DAMSPP"] - df_all["RTMSPP"]
)

df_all["NodeType"] = df_all["AssetID"] + "_" + df_all["TransactionType"]
df_all["DateTime_hr"] = df_all["DateTime"].dt.normalize() + pd.to_timedelta(df_all["Hour"], unit="h")

pivot_all = df_all.pivot_table(index="DateTime", columns=["AssetID", "Hour"], values="spread").sort_index()


modelos_info = []
for carpeta in os.listdir("modelos_guardados"):
    if "_h" not in carpeta:
        continue
    asset_part, rest = carpeta.split("_h")
    hour_str, tx_type = rest.split("_", 1)
    hour = int(hour_str)
    path = f"modelos_guardados/{carpeta}"

    modelos_info.append({
        "asset_id": asset_part,
        "hour": hour,
        "tx_type": tx_type.upper(),
        "dir": path,
        "features": joblib.load(f"{path}/features.joblib"),
        "valid_mask": joblib.load(f"{path}/valid_mask.joblib"),
        "external_features": joblib.load(f"{path}/external_features.joblib"),
        "best_params": joblib.load(f"{path}/best_params.joblib"),
        "raw": pd.read_csv(f"{path}/raw.csv", parse_dates=["DateTime"]),
        "dates": [], "preds": [], "actuals": []
    })


test_start = max(
    m["raw"]["DateTime"].min().normalize() + pd.DateOffset(months=LOOKBACK_MONTHS)
    for m in modelos_info
) + timedelta(days=D_LAG)

test_end = min(m["raw"]["DateTime"].max().normalize() for m in modelos_info)
current_day = test_start

print(f"Back-test desde {test_start.date()} hasta {test_end.date()}")

resultados_diarios = []
nodos_por_dia = {}

asignaciones_all = []

while current_day <= test_end:
    start = time.time()
    print(f"\n=== Día {current_day.date()} ===========================")
    dia_preds = []

    for model in modelos_info:
        raw = model["raw"]

        feat_cols = model["features"]
        external_features = model["external_features"]
        valid_mask = model["valid_mask"]
        best_params = model["best_params"]

        hist_cut = current_day - timedelta(days=D_LAG)
        end_hist = hist_cut

        hist_df = raw[raw["DateTime"] <= hist_cut]
        pivot_cut = pivot_all.loc[:hist_cut]

        feat_hist, _ = make_features(hist_df, pivot_cut, external_features, valid_mask)
        feat_hist = feat_hist.dropna(subset=feat_cols + ["spread"])
        if feat_hist.empty or len(feat_hist) < 50:
            continue

        model_xgb = XGBRegressor(**best_params, random_state=RANDOM_STATE)
        model_xgb.fit(feat_hist[feat_cols], feat_hist["spread"])

        day_row = raw[
            (raw["DateTime"].dt.normalize() == current_day) &
            (raw["Hour"] == model["hour"])
        ]
        if day_row.empty:
            continue
        pivot_day = pivot_all.loc[:end_hist]
        aux_df = pd.concat([hist_df, day_row])

        feat_day, _ = make_features(aux_df, pivot_cut, external_features, valid_mask)
        X_day = feat_day.loc[feat_day["DateTime"].dt.normalize() == current_day, feat_cols]
        if X_day.empty or X_day.isna().any().any():
            continue

        y_pred = model_xgb.predict(X_day)[0]

        y_real = day_row["spread"].values[0]

        model["dates"].append(current_day)
        model["preds"].append(y_pred)
        model["actuals"].append(y_real)

        errors_hist = np.abs(np.array(model["preds"]) - np.array(model["actuals"]))
        if len(errors_hist) >= 5:
            w_err = np.dot(PESOS_ERR, errors_hist[-5:-2]) / PESOS_ERR.sum()

        else:
            w_err = np.inf

        dia_preds.append({
            "asset_id": model["asset_id"],
            "hour": model["hour"],
            "tipo": model["tx_type"],
            "y_pred": y_pred,
            "y_real": y_real,
            "dam": day_row["DAMSPP"].values[0],
            "rtm": day_row["RTMSPP"].values[0],
            "w_err": w_err
        })

        print(f"[Predicción] {model['asset_id']} h{model['hour']} ({model['tx_type']}): "
              f"Pred = {y_pred:.2f}, Real = {y_real:.2f}, "
              f"w_err = {w_err:.2f},"
              f"DAM = {day_row['DAMSPP'].values[0]:.2f}, RTM = {day_row['RTMSPP'].values[0]:.2f}")

    nodo_preds = defaultdict(list)
    for d in dia_preds:
        nodo_preds[d["asset_id"]].append(d)

    mejores_por_nodo = [
        max(ops, key=lambda x: x["y_pred"] / (x["w_err"] + 1e-6))
        for ops in nodo_preds.values()
    ]

    diversificados = []
    capital_usado = 0
    nodos_utilizados = set()
    nodos_hora_utilizados = set()


    for d in sorted(mejores_por_nodo, key=lambda x: x["y_pred"] / (x["w_err"] + 1e-6), reverse=True):
        if (d["asset_id"], d["hour"]) in nodos_hora_utilizados:
            continue
        if d["y_pred"] <= 0 or not np.isfinite(d["w_err"]) or d["w_err"] > MAX_WERR:
            continue

        precio_compra_hist = get_dam_hist(raw, current_day, d["hour"], offset=2)
        if np.isnan(precio_compra_hist) or precio_compra_hist <= 0:
            continue
        mw = int(CAPITAL_POR_NODO / precio_compra_hist)
        if mw == 0:
            continue
        if mw > MW_MAXIMO:
            continue
        retorno_esperado = d["y_pred"] * mw
        if retorno_esperado < MIN_EXPECTED_PROFIT:
            continue
        costo_tx = precio_compra_hist * mw
        if (capital_usado + costo_tx) > CAPITAL_TOTAL_DIA:
            continue

        d["MW"] = mw
        d["costo_tx"] = costo_tx
        diversificados.append(d)
        capital_usado += costo_tx
        nodos_utilizados.add(d["asset_id"])
        nodos_hora_utilizados.add((d["asset_id"], d["hour"]))

        if len(nodos_utilizados) >= MIN_NODOS_DIA:
            break

    restantes = sorted(
        [x for x in dia_preds if (x["asset_id"], x["hour"]) not in nodos_hora_utilizados],
        key=lambda x: x["y_pred"] / (x["w_err"] + 1e-6),
        reverse=True
    )

    rentables = diversificados.copy()


    for d in restantes:
        if (d["asset_id"], d["hour"]) in nodos_hora_utilizados:
            continue
        if d["y_pred"] <= 0 or not np.isfinite(d["w_err"]) or d["w_err"] > MAX_WERR:
            continue

        precio_compra_hist = get_dam_hist(raw, current_day, d["hour"], offset=2)
        if np.isnan(precio_compra_hist) or precio_compra_hist <= 0:
            continue
        mw = int(CAPITAL_POR_NODO / precio_compra_hist)
        if mw == 0:
            continue
        if mw > MW_MAXIMO:
            continue            
        retorno_esperado = d["y_pred"] * mw
        if retorno_esperado < MIN_EXPECTED_PROFIT:
            continue
        costo_tx = precio_compra_hist * mw
        if (capital_usado + costo_tx) > CAPITAL_TOTAL_DIA:
            continue

        d["MW"] = mw
        d["costo_tx"] = costo_tx
        rentables.append(d)
        capital_usado += costo_tx
        nodos_utilizados.add(d["asset_id"])
        nodos_hora_utilizados.add((d["asset_id"], d["hour"]))

    if len(nodos_utilizados) < MIN_NODOS_DIA:
        print(f" Día {current_day.date()}: solo se asignaron {len(nodos_utilizados)} nodos únicos.")


    if len(nodos_utilizados) < MIN_NODOS_DIA:
        print(f" Día {current_day.date()}: forzando inclusión de más nodos para cumplir mínimo de {MIN_NODOS_DIA}")
        candidatos_extra = [
            d for d in dia_preds
            if (d["asset_id"], d["hour"]) not in nodos_hora_utilizados
            and d["asset_id"] not in nodos_utilizados
            and d["y_pred"] > 0
            and np.isfinite(d["w_err"]) and d["w_err"] <= MAX_WERR * 2
        ]
        candidatos_extra = sorted(candidatos_extra, key=lambda x: x["y_pred"] / (x["w_err"] + 1e-6), reverse=True)

        for d in candidatos_extra:
            precio_compra_hist = get_dam_hist(raw, current_day, d["hour"], offset=2)
            if np.isnan(precio_compra_hist) or precio_compra_hist <= 0:
                continue
            mw = int(CAPITAL_POR_NODO / precio_compra_hist)
            if mw == 0:
                continue
            if mw > MW_MAXIMO:
                continue                
            costo_tx = mw * precio_compra_hist
            if (capital_usado + costo_tx) > CAPITAL_TOTAL_DIA:
                continue

            d["MW"] = mw
            d["costo_tx"] = costo_tx
            rentables.append(d)
            capital_usado += costo_tx
            nodos_utilizados.add(d["asset_id"])
            nodos_hora_utilizados.add((d["asset_id"], d["hour"]))
            print(f" Nodo forzado: {d['asset_id']} h{d['hour']} ({d['tipo']})")

            if len(nodos_utilizados) >= MIN_NODOS_DIA:
                break

    retorno_dia = 0
    for nodo in rentables:
        costo = nodo["dam"] if nodo["tipo"] == "BID" else nodo["rtm"]
        ingreso = nodo["rtm"] if nodo["tipo"] == "BID" else nodo["dam"]
        ganancia = (ingreso - costo) * nodo["MW"]
        retorno_dia += ganancia

        print(f"{nodo['asset_id']} h{nodo['hour']} ({nodo['tipo']}): MW = {nodo['MW']}, Retorno = {ganancia:.2f}")

        asignaciones_all.append({
            "DateTime_hr": current_day + timedelta(hours=nodo["hour"]),
            "AssetID": nodo["asset_id"],
            "TransactionType": nodo["tipo"],
            "MW": nodo["MW"],
            "DAMspp": nodo["dam"],
            "RTMspp": nodo["rtm"],
            "SpreadPred": nodo["y_pred"],
            "SpreadReal": nodo["y_real"],
            "Costo_tx_USD": nodo["costo_tx"]
        })

    resultados_diarios.append({
        "fecha": current_day.date(),
        "retorno_total": retorno_dia,
        "n_operaciones": len(rentables)
    })

    nodos_por_dia[current_day.date()] = rentables.copy()

    print(f"Resultado día {current_day.date()}: retorno_total = {retorno_dia:.2f}, operaciones = {len(rentables)}")
    print(f"Tiempo transcurrido: {time.time() - start:.2f} segundos")

    current_day += timedelta(days=1)


df_retornos = pd.DataFrame(resultados_diarios)
df_retornos["retorno_acumulado"] = df_retornos["retorno_total"].cumsum()
df_retornos["RoI"] = df_retornos["retorno_total"] / (CAPITAL_POR_NODO * TOP_N_SPREADS)


total_roi = df_retornos["retorno_total"].sum() / (CAPITAL_POR_NODO * len(df_retornos))
print(f"\n RoI total: {total_roi:.4f}")


df_retornos["peak"] = df_retornos["retorno_acumulado"].cummax()
df_retornos["drawdown"] = df_retornos["retorno_acumulado"] - df_retornos["peak"]
max_drawdown = df_retornos["drawdown"].min()
print(f" Máximo Drawdown: {max_drawdown:.2f}")


sharpe_ratio = df_retornos["RoI"].mean() / df_retornos["RoI"].std()
print(f"Sharpe Ratio (aprox.): {sharpe_ratio:.2f}")


df_retornos.to_csv("logs/retornos_diarios.csv", index=False)

plt.figure(figsize=(10, 4))
plt.plot(df_retornos["fecha"], df_retornos["retorno_acumulado"], label="Retorno acumulado")
plt.grid()
plt.title("Retorno acumulado BID + OFFER")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("logs/retorno_acumulado.png")
plt.close()


df_retornos = pd.DataFrame(resultados_diarios)
df_retornos["retorno_acumulado"] = df_retornos["retorno_total"].cumsum()
df_retornos.to_csv("logs/retornos_diarios.csv", index=False)


ranking = []

for fecha, nodos in nodos_por_dia.items():
    for nodo in nodos:
        costo = nodo["dam"] if nodo["tipo"] == "BID" else nodo["rtm"]
        ingreso = nodo["rtm"] if nodo["tipo"] == "BID" else nodo["dam"]
        retorno = (ingreso - costo) * nodo["MW"]

        ranking.append({
            "fecha": fecha,
            "asset_id": nodo["asset_id"],
            "hour": nodo["hour"],
            "tipo": nodo["tipo"],
            "retorno": retorno
        })

df_ranking = pd.DataFrame(ranking)
df_nodos = (
    df_ranking.groupby(["asset_id", "hour", "tipo"])
    .agg(
        veces_seleccionado=("retorno", "count"),
        retorno_total=("retorno", "sum"),
        retorno_promedio=("retorno", "mean")
    )
    .sort_values(by="retorno_total", ascending=False)
    .reset_index()
)

df_nodos.to_csv("logs/ranking_nodos.csv", index=False)

print("\n Top 10 nodos más rentables:")
print(df_nodos.head(10).to_string(index=False))



for model in modelos_info:
    df_preds = pd.DataFrame({
        "Date": model["dates"],
        "SpreadReal": model["actuals"],
        "SpreadPred": model["preds"]
    })
    name = f"{model['asset_id']}_h{model['hour']}_{model['tx_type']}"
    df_preds.to_csv(f"logs/predicciones_{name}.csv", index=False)


df_asig = pd.DataFrame(asignaciones_all)
df_asig.to_csv("logs/asignaciones_horarias.csv", index=False)
print(f" Dataset horario de asignaciones guardado: {len(df_asig)} filas")


print("\n Walkforward finalizado. Archivos guardados en carpeta 'logs'.")
