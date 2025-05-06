# Predicción de Spreads en el Mercado Eléctrico

Este proyecto consiste en un sistema para predecir y asignar capital diariamente en mercados eléctricos nodales, utilizando aprendizaje supervisado. Emplea modelos de regresión XGBoost con hiperparámetros optimizados mediante Optuna, utilizando precios horarios de los mercados DAM y RTM.

El objetivo es maximizar el retorno económico diario, cumpliendo restricciones como presupuesto fijo, diversificación mínima de nodos, disponibilidad rezagada de datos (dos días) y manejo adecuado de datos incompletos. El desempeño se evalúa mediante validación tipo walk-forward.

El proyecto incluye:

- Preprocesamiento y limpieza del conjunto de datos original.
- Construcción de variables predictivas internas y externas.
- Entrenamiento individual de modelos para cada combinación nodo-hora-tipo de transacción.
- Simulación diaria para decidir asignaciones de capital y evaluar métricas como ROI, drawdown y Sharpe ratio.

## Estructura del Proyecto

- `PreprocesamientoDataSet.py`: Limpieza, normalización y enriquecimiento del dataset.
- `utils_features.py`: Generación de características (features) y cálculo de correlaciones con spreads rezagados.
- `EntrenamientoModelosT0.py`: Entrenamiento de modelos `XGBoost` para diferentes nodos y horas.
- `Walkforward_Market.py`: Evaluación de los modelos entrenados en un esquema walkforward con simulación de decisiones de compra/venta.

---

## Requisitos

- Python 3.8+
- Paquetes:

```bash
pip install -r requirements.txt
```

---

## Instrucciones de Ejecución

### 1. Preprocesamiento del Dataset

Asegúrate de tener el archivo original `dataset_pt_20250428v0.csv` en el mismo directorio. Luego ejecuta:

```bash
python PreprocesamientoDataSet.py
```

Esto generará `dataset_clean.csv` y guardará archivos auxiliares en la carpeta `logs`.

### 2. Entrenamiento de Modelos

Entrena modelos para cada nodo y hora seleccionados:

```bash
python EntrenamientoModelosT0.py
```

Los modelos se guardan en la carpeta `modelos_guardados/`.

### 3. Evaluación Walkforward

Simula la estrategia de mercado usando los modelos entrenados:

```bash
python Walkforward_Market.py
```

Resultados, métricas y gráficos se guardarán en la carpeta `logs/`.

---

## Salidas Generadas

- `dataset_clean.csv`: Dataset preprocesado.
- `modelos_guardados/`: Modelos entrenados por nodo y hora.
- `logs/`: Reportes, gráficos y CSVs con resultados y predicciones.

---

## Notas

- Los scripts generan automáticamente las carpetas necesarias (`logs`, `modelos_guardados`) si no existen.
- Se utilizan técnicas de imputación, rolling windows y validación temporal.

---

## Autor

Este proyecto fue desarrollado como parte de una solución para predicción de spreads de precios en mercados eléctricos.


---

## Scripts Adicionales

En la subcarpeta `VolatilityPredictionModel` se encuentra el archivo `VolatilityPredictionModel.py`. Este script no forma parte del flujo principal del sistema, sino que corresponde a una recomendación de mejora para incluir modelos que predicen la volatilidad del spread como medida adicional de riesgo.