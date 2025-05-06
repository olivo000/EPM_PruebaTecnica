# Predicción de Spreads en el Mercado Eléctrico

Este proyecto realiza el preprocesamiento, entrenamiento de modelos y evaluación de estrategias para predecir el spread de precios entre los mercados DAM y RTM en el sector eléctrico mexicano.

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


---

