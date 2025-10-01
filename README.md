
# COI/Fraud mensual + NLP MX + Seaborn Viz + Q&A

## Requisitos
- pandas, numpy, seaborn, scikit-learn, scipy (opcional para embeddings)

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

df = pd.read_csv("/content/mis_transacciones.csv")  # columnas: persona_1, persona_2, relacion, fecha_hora, monto, descripcion
reports = run_pipeline(df)

# gráficos
plots.plot_person_imbalance_bar(reports)
```

## CLI
```bash
python -m coi_fraud --csv ./mis_transacciones.csv --out ./forensic_outputs
```
