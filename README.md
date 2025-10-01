# COI/Fraud mensual + NLP MX + Seaborn Viz + Q&A

## Requisitos
- Instalar dependencias con `pip install -r requirements.txt` (incluye pandas, numpy, seaborn, scikit-learn y scipy; estas dos últimas son opcionales si no se entrenan embeddings)

## Instalación en Google Colab
1. Conéctate a un cuaderno nuevo y asegúrate de estar usando un entorno con Python 3.10+.
2. Descarga el paquete (por ejemplo, clonando el repositorio o subiendo el archivo `.zip` generado por GitHub/tu release) y descomprímelo dentro de `/content`. Un flujo típico sería:
   ```python
   !git clone https://github.com/tu-org/coi.git /content/coi
   %cd /content/coi
   ```
   Si prefieres trabajar con un archivo `.zip`, basta con subirlo a Colab (o montarlo desde Drive) y ejecutar:
   ```python
   from zipfile import ZipFile

   with ZipFile("/content/coi_fraud_mensual_viz_qa.zip", "r") as z:
       z.extractall("/content")
   ```
3. Instala las dependencias mínimas:
   ```python
   %pip install -q pandas numpy seaborn scikit-learn scipy
   ```
4. Añade el paquete al `sys.path` si no usas instalación editable. Suponiendo que el módulo vive en `/content/coi_fraud`:
   ```python
   import sys
   sys.path.append("/content")  # o "/content/coi" si clonaste el repositorio
   ```
5. Verifica la importación con `import coi_fraud`.

> 💡 Consejo: si montas Google Drive (`from google.colab import drive; drive.mount("/content/drive")`), puedes almacenar datasets voluminosos y leerlos directamente con `pd.read_csv("/content/drive/.../archivo.csv")`.

## Ejemplos de uso
### 1. Pipeline completo con pandas
```python
import pandas as pd
from coi_fraud import run_pipeline, generate_diverse_dataset

# Dataset mínimo de ejemplo
raw = pd.DataFrame(
    {
        "user_id": ["A", "A", "B", "C"],
        "receptor-user_id": ["B", "C", "A", "A"],
        "load_date": ["2024-01-05", "2024-01-17", "2024-01-09", "2024-01-22"],
        "movement_amount": [1200, 850, 1200, 5000],
        "transaction_desc": [
            "Pago de servicios",
            "Bonificación extraordinaria",
            "Reembolso gasto",
            "Transferencia gerencia",
        ],
    }
)

reports = run_pipeline(raw)

# Acceder al resumen de personas del último mes
personas_mes = reports["persona"]["ultimo_mes"]
print(personas_mes[["persona", "risk_avg_person", "sum_emit", "sum_recv"]].head())
```

### 2. Análisis guiado con el módulo de Q&A
```python
from coi_fraud.analysis import qa

# Personas con mayor desbalance neto emite-recibe (todo el periodo)
desbalance = qa.desbalance_personas(reports)
print(desbalance[["persona", "net", "ratio" ]].head())

# Pairs con comportamiento Yo-Yo consecutivo
yo_yo = qa.yoyo_consecutivos(reports, timeframe="ultimos_3_meses")
print(yo_yo.head())
```

### 3. Visualizaciones rápidas con Seaborn/Matplotlib
```python
import matplotlib.pyplot as plt
from coi_fraud.viz import plots

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plots.plot_person_imbalance_bar(reports, timeframe="todo_el_tiempo", ax=axes[0], top_n=10)
plots.plot_manager_concepts_bar(reports, timeframe="todo_el_tiempo", ax=axes[1])
plt.tight_layout()
plt.show()
```

### 4. Exportar reportes a CSV
Los objetos devueltos por `run_pipeline` son dataframes almacenados en un diccionario. Puedes serializarlos fácilmente:
```python
for categoria, timeframes in reports.items():
    for periodo, df in timeframes.items():
        ruta = f"/content/salida_{categoria}_{periodo}.csv"
        df.to_csv(ruta, index=False)
```

## Uso rápido (Colab)
```python
!pip -q install pandas numpy seaborn scikit-learn scipy
from zipfile import ZipFile
with ZipFile("/content/coi_fraud_mensual_viz_qa.zip", "r") as z:
    z.extractall("/content")
import sys; sys.path.append("/content")
import pandas as pd
from coi_fraud import run_pipeline
from coi_fraud.viz import plots
from coi_fraud.analysis import qa

df = pd.read_csv("/content/mis_transacciones.csv")  # columnas mínimas: user_id, receptor-user_id, load_date, movement_amount, transaction_desc
reports = run_pipeline(df)

# gráficos
plots.plot_person_imbalance_bar(reports)
```

### 5. Generar un dataset de prueba diverso
```python
from coi_fraud import generate_diverse_dataset

dataset = generate_diverse_dataset()  # 10 000 filas por defecto
```

## CLI
```bash
python -m coi_fraud --csv ./mis_transacciones.csv --out ./forensic_outputs
```
