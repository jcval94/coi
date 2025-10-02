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

## Preguntas de experimentación (Q1–Q7)
El módulo `experiment_questions.py` genera respuestas tabulares para siete preguntas recurrentes a partir del diccionario de `reports` producido por `run_pipeline`. Todas las funciones aceptan `timeframe` (por defecto `"todo_el_tiempo"`) y devuelven columnas de interpretabilidad en español listando la lógica aplicada.

- **Q1 – Manager con conceptos NLP sospechosos** (`question1_manager_nlp`):
  - **Metodología:** filtra transacciones manager-subordinado en `reports["transaccion"][timeframe]`, concatena campos `nlp_concepto_sospechoso`, `descripcion` y `tx_tags`, y ejecuta coincidencias por expresiones regulares contra las categorías `("SOBORNO", "FACILITACIÓN", "OFUSCACIÓN", "EXTORSIÓN", "FAVORES SEXUALES")` y sus sinónimos (`NLP_CATEGORY_SYNONYMS`). Agrupa por mes, categoría detectada, manager y subordinado para resumir `tx_count` y `monto_total` y generar textos explicativos.
  - **Parámetros clave:** `timeframe`; `categories` (lista de categorías NLP, por defecto las cinco anteriores).
  - **Visualización rápida:**
    ```python
    import matplotlib.pyplot as plt
    from experiment_visualizations import plot_q1_manager_nlp

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_q1_manager_nlp(reports, timeframe="todo_el_tiempo", ax=ax)
    plt.tight_layout()
    plt.show()
    ```

- **Q2 – Conceptos NLP con mayor severidad** (`question2_manager_concepts`):
  - **Metodología:** reutiliza la detección de Q1 sobre `reports["transaccion"][timeframe]` y agrega por mes y categoría calculando número de transacciones y el cuantil 0.95 de `risk_score`, ordenando por severidad antes de redactar la explicación.
  - **Parámetros clave:** `timeframe`; categorías NLP internas (idénticas a Q1, sin exponer otro parámetro).
  - **Visualización rápida:**
    ```python
    import matplotlib.pyplot as plt
    from experiment_visualizations import plot_q2_manager_concepts

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_q2_manager_concepts(reports, timeframe="todo_el_tiempo", ax=ax)
    plt.tight_layout()
    plt.show()
    ```

- **Q3 – Pares con rasgos Quid Pro Quo** (`question3_quid_pairs`):
  - **Metodología:** parte del resumen `reports["casuistica_quid_pro_quo_par"][timeframe]` y del detalle `reports["casuistica_quid_pro_quo_tx"][timeframe]` (o de `transaccion` si faltan). Selecciona pares con `quid_score_max ≥ min_score` (2.2 por defecto), proporción de interacciones jerárquicas (`quid_manager_ratio`) superior a `min_manager_ratio` (0.5) y alguna aprobación o compensación. Si no hay resultados, recurre a las transacciones base calculando agregados por par, con un modo relajado que prioriza los puntajes más altos disponibles. Complementa con un listado de transacciones destacadas por `feat_quid_score` y, si fue necesario, con umbrales relajados.
  - **Parámetros clave:** `timeframe`; `min_score` (float, 2.2 por defecto); `min_manager_ratio` (float, 0.5 por defecto).
  - **Visualización rápida:**
    ```python
    import matplotlib.pyplot as plt
    from experiment_visualizations import plot_q3_quid_pairs

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_q3_quid_pairs(reports, timeframe="todo_el_tiempo", ax=ax)
    plt.tight_layout()
    plt.show()
    ```

- **Q4 – Autorizaciones con valor negativo vs. carga** (`question4_quid_negative_value_vs_load`):
  - **Metodología:** inspecciona `casuistica_quid_pro_quo_tx` (o `transaccion` como respaldo) buscando transacciones con `feat_quid_value_vs_load_days < 0`. Si no existen, selecciona las 10 con menores desfases registrados o, en su defecto, los mayores puntajes `feat_quid_score`. Calcula el posible responsable según la relación jerárquica (`relacion`/`feat_quid_rel_label`) y redacta el resumen indicando desfase, puntaje y responsable identificado; marca explícitamente cuando se usó el criterio relajado.
  - **Parámetros clave:** `timeframe`.
  - **Visualización rápida:**
    ```python
    import matplotlib.pyplot as plt
    from experiment_visualizations import plot_q4_negative_value_vs_load

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_q4_negative_value_vs_load(reports, timeframe="todo_el_tiempo", ax=ax)
    plt.tight_layout()
    plt.show()
    ```

- **Q5 – Reutilización de referencias de pago** (`question5_reference_reuse`):
  - **Metodología:** trabaja con `casuistica_referencia_resumen` y `casuistica_referencia_tx`. Prioriza referencias que aparezcan en más de un par (`n_pairs > 1`), ordenadas por número de pares, rango de días (`days_range`) y transacciones. Si esos datos no existen, reconstruye la métrica desde `transaccion` usando `feat_reference_norm` (o normalizando `descripcion`), calcula métricas temporales y filtra por reutilización en ≤30 días; de no haber candidatos, activa un modo relajado que lista las referencias más frecuentes. Finalmente, detalla las transacciones asociadas a cada referencia recurrente.
  - **Parámetros clave:** `timeframe` (el resto de umbrales y normalizaciones están codificados en la función; no se exponen parámetros adicionales).
  - **Visualización rápida:**
    ```python
    import matplotlib.pyplot as plt
    from experiment_visualizations import plot_q5_reference_reuse

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_q5_reference_reuse(reports, timeframe="todo_el_tiempo", ax=ax)
    plt.tight_layout()
    plt.show()
    ```

- **Q6 – Receptores centralizadores** (`question6_centralizers`):
  - **Metodología:** resume `reports["transaccion"][timeframe]` por mes y receptor (`receptor-user_id`), calculando el `inflow` total, emisores únicos, número de transacciones y riesgo promedio (`risk_score`). Define una métrica de `centralidad = inflow * emisores_unicos`, ordena de mayor a menor por mes y genera explicaciones con esos indicadores.
  - **Parámetros clave:** `timeframe`.
  - **Visualización rápida:**
    ```python
    import matplotlib.pyplot as plt
    from experiment_visualizations import plot_q6_centralizers

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_q6_centralizers(reports, timeframe="todo_el_tiempo", ax=ax)
    plt.tight_layout()
    plt.show()
    ```

- **Q7 – Personas con desbalance neto** (`question7_net_imbalance`):
  - **Metodología:** toma `reports["persona"][timeframe]`, asegura la presencia de `desbalance_persona_monto_neto`, calcula su valor absoluto para ordenar y conserva también los contadores de meses extremos enviando y recibiendo. Produce interpretabilidad destacando el desbalance monetario y los meses con comportamiento extremo.
  - **Parámetros clave:** `timeframe`.
  - **Visualización rápida:**
    ```python
    import matplotlib.pyplot as plt
    from experiment_visualizations import plot_q7_net_imbalance

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_q7_net_imbalance(reports, timeframe="todo_el_tiempo", ax=ax)
    plt.tight_layout()
    plt.show()
    ```

- **Q8 – Receptores nuevos con montos altos** (`question8_case13_new_employees`):
  - **Metodología:** utiliza `reports["persona"][timeframe]` para filtrar a receptores con bandera `caso13_persona_flag_nuevo_receptor_altos_montos`. Prioriza a quienes recibieron montos altos (percentil 90) dentro de sus primeras interacciones (≤6 meses) y calcula totales, emisores únicos y promedios.
  - **Parámetros clave:** `timeframe`.
  - **Ejemplo de uso:**
    ```python
    from experiment_questions import question8_case13_new_employees

    q8 = question8_case13_new_employees(reports, timeframe="todo_el_tiempo")
    print(q8.head())
    ```
  - **Visualización rápida:**
    ```python
    import matplotlib.pyplot as plt
    from experiment_visualizations import plot_q8_case13_new_employees

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_q8_case13_new_employees(reports, timeframe="todo_el_tiempo", ax=ax)
    plt.tight_layout()
    plt.show()
    ```

- **Q9 – Veteranos que reciben de emisores nuevos** (`question9_case14_veterans_from_newcomers`):
  - **Metodología:** revisa `reports["persona"][timeframe]` buscando la bandera `caso14_persona_flag_antiguo_recibe_de_nuevos` o, en su defecto, reconstruye las métricas desde transacciones y heurísticas de antigüedad. Agrega transacciones y montos recibidos de emisores recientes resaltando emisores únicos y promedios.
  - **Parámetros clave:** `timeframe`.
  - **Ejemplo de uso:**
    ```python
    from experiment_questions import question9_case14_veterans_from_newcomers

    q9 = question9_case14_veterans_from_newcomers(reports, timeframe="todo_el_tiempo")
    print(q9.head())
    ```
  - **Visualización rápida:**
    ```python
    import matplotlib.pyplot as plt
    from experiment_visualizations import plot_q9_case14_veterans_from_newcomers

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_q9_case14_veterans_from_newcomers(reports, timeframe="todo_el_tiempo", ax=ax)
    plt.tight_layout()
    plt.show()
    ```

- **Q10 – Rachas Yo-Yo prolongadas** (`question10_yoyo_streaks`):
  - **Metodología:** analiza `reports["transaccion"][timeframe]` con la bandera `sig_yoyo` para encontrar pares bidireccionales con rachas consecutivas de ida y vuelta. Si faltan banderas, aplica heurísticas de ventanas horarias. Cruza con el resumen `par_personas` para incorporar riesgo y filtra por rachas mínimas y riesgo máximo.
  - **Parámetros clave:** `timeframe`; `min_consecutive` (mínimo de eventos consecutivos, por defecto `2`); `risk_threshold` (riesgo mínimo del par, por defecto `1.8`).
  - **Ejemplo de uso:**
    ```python
    from experiment_questions import question10_yoyo_streaks

    q10 = question10_yoyo_streaks(reports, timeframe="ultimos_3_meses", min_consecutive=3)
    print(q10[["par_bidir", "racha_max_yo_yo", "riesgo_max_par"]].head())
    ```
  - **Visualización rápida:**
    ```python
    import matplotlib.pyplot as plt
    from experiment_visualizations import plot_q10_yoyo_streaks

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_q10_yoyo_streaks(reports, timeframe="todo_el_tiempo", ax=ax)
    plt.tight_layout()
    plt.show()
    ```

- **Q11 – Montos pegados a umbrales regulatorios** (`question11_near_threshold_structuring`):
  - **Metodología:** procesa `reports["transaccion"][timeframe]` para identificar pares con bandera `sig_near_thr` y deltas pequeños (`feat_delta_near_thr`) respecto a umbrales comunes. Si falta la métrica, estima la distancia a umbrales típicos y aplica heurísticas flexibles para garantizar cobertura. Resume meses con recurrencia, monto total y riesgo.
  - **Parámetros clave:** `timeframe`; `min_months` (meses mínimos con recurrencia, por defecto `3`); `delta_limit` (diferencia máxima al umbral, por defecto `10.0`).
  - **Ejemplo de uso:**
    ```python
    from experiment_questions import question11_near_threshold_structuring

    q11 = question11_near_threshold_structuring(reports, timeframe="todo_el_tiempo", min_months=2)
    print(q11[["pair", "meses_con_near", "monto_total_near"]].head())
    ```
  - **Visualización rápida:**
    ```python
    import matplotlib.pyplot as plt
    from experiment_visualizations import plot_q11_near_threshold_structuring

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_q11_near_threshold_structuring(reports, timeframe="todo_el_tiempo", ax=ax)
    plt.tight_layout()
    plt.show()
    ```

- **Q12 – Smurfing crónico** (`question12_smurfing_chronic`):
  - **Metodología:** parte de `reports["transaccion"][timeframe]` y la bandera `sig_smurf` para localizar pares con depósitos fragmentados pequeños a lo largo de varios meses. Si no hay banderas, usa cuantiles por par para etiquetar montos reducidos y estima tendencias de riesgo promedio y máximo por mes.
  - **Parámetros clave:** `timeframe`; `min_months` (meses mínimos con smurfing, por defecto `3`).
  - **Ejemplo de uso:**
    ```python
    from experiment_questions import question12_smurfing_chronic

    q12 = question12_smurfing_chronic(reports, timeframe="todo_el_tiempo", min_months=4)
    print(q12[["pair", "meses_con_smurf", "monto_smurf_total"]].head())
    ```
  - **Visualización rápida:**
    ```python
    import matplotlib.pyplot as plt
    from experiment_visualizations import plot_q12_smurfing_chronic

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_q12_smurfing_chronic(reports, timeframe="todo_el_tiempo", ax=ax)
    plt.tight_layout()
    plt.show()
    ```

- **Q13 – Préstamos incumplidos con ráfagas de frecuencia** (`question13_bad_loans_with_frequency`):
  - **Metodología:** inspecciona banderas `sig_loan_bad_repay` y `sig_freq` dentro de `reports["transaccion"][timeframe]`. Calcula coincidencias mensuales de préstamos con repago ≤50% y eventos de alta frecuencia; cuando faltan banderas, emplea heurísticas bidireccionales para estimar préstamos, reembolsos y umbrales de frecuencia. Agrega meses coincidentes, montos y riesgos.
  - **Parámetros clave:** `timeframe`; `min_months` (meses mínimos de coincidencia, por defecto `3`).
  - **Ejemplo de uso:**
    ```python
    from experiment_questions import question13_bad_loans_with_frequency

    q13 = question13_bad_loans_with_frequency(reports, timeframe="todo_el_tiempo")
    print(q13[["pair", "meses_con_coincidencia", "monto_prestamos_incumplidos"]].head())
    ```
  - **Visualización rápida:**
    ```python
    import matplotlib.pyplot as plt
    from experiment_visualizations import plot_q13_bad_loans_with_frequency

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_q13_bad_loans_with_frequency(reports, timeframe="todo_el_tiempo", ax=ax)
    plt.tight_layout()
    plt.show()
    ```

- **Q14 – Pagos recurrentes tipo nómina** (`question14_recurrent_payroll`):
  - **Metodología:** utiliza `reports["transaccion"][timeframe]` y la bandera `sig_recurrent` para agrupar pagos emitidos cerca de un mismo día de corte. Identifica meses consecutivos con comportamiento recurrente y calcula totales, promedios y número de pagos, relajando banderas cuando es necesario para mantener cobertura.
  - **Parámetros clave:** `timeframe`; `min_months` (meses consecutivos mínimos, por defecto `3`).
  - **Ejemplo de uso:**
    ```python
    from experiment_questions import question14_recurrent_payroll

    q14 = question14_recurrent_payroll(reports, timeframe="todo_el_tiempo", min_months=4)
    print(q14[["emisor", "receptor", "meses_recurrentes", "monto_total"]].head())
    ```
  - **Visualización rápida:**
    ```python
    import matplotlib.pyplot as plt
    from experiment_visualizations import plot_q14_recurrent_payroll

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_q14_recurrent_payroll(reports, timeframe="todo_el_tiempo", ax=ax)
    plt.tight_layout()
    plt.show()
    ```

- **Q15 – Clusters con señales coordinadas** (`question15_coordinated_cluster_signals`):
  - **Metodología:** revisa `reports["clusters_personas"][timeframe]` y consolida las tasas de señales priorizadas (yo-yo, smurf, ciclos, quid y referencias reutilizadas). Normaliza conteos y montos para ordenar los clusters por número de señales activas, riesgo máximo y volumen total, generando un texto que resalta la persona más desbalanceada y el detalle porcentual de cada señal.
  - **Parámetros clave:** `timeframe`; `top_n` (número máximo de clusters a devolver, 10 por defecto en la función base).
  - **Ejemplo de uso:**
    ```python
    from experiment_questions import question15_coordinated_cluster_signals

    q15 = question15_coordinated_cluster_signals(reports, timeframe="todo_el_tiempo", top_n=5)
    print(q15[["cluster_id", "signals_activas", "interpretabilidad"]].head())
    ```
  - **Visualización rápida:**
    ```python
    import matplotlib.pyplot as plt
    from experiment_visualizations import plot_q15_coordinated_cluster_signals

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_q15_coordinated_cluster_signals(reports, timeframe="todo_el_tiempo", ax=ax)
    plt.tight_layout()
    plt.show()
    ```

- **Q16 – Transacciones con múltiples señales simultáneas** (`question16_multisignal_transactions`):
  - **Metodología:** analiza `reports["transaccion"][timeframe]` buscando operaciones que acumulen varias banderas (jerarquía, yo-yo, smurf, near-threshold, quid y cambio brusco). Prioriza las que alcanzan al menos tres señales activas, relajando el umbral si no hay suficientes resultados, y redacta interpretabilidad con monto, riesgo, relación declarada y descripción.
  - **Parámetros clave:** `timeframe`; `top_n` (máximo de transacciones listadas, 25 por defecto).
  - **Ejemplo de uso:**
    ```python
    from experiment_questions import question16_multisignal_transactions

    q16 = question16_multisignal_transactions(reports, timeframe="todo_el_tiempo", top_n=10)
    print(q16[["fecha_hora_ts", "emisor", "receptor", "signals_detalle"]].head())
    ```
  - **Visualización rápida:**
    ```python
    import matplotlib.pyplot as plt
    from experiment_visualizations import plot_q16_multisignal_transactions

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_q16_multisignal_transactions(reports, timeframe="todo_el_tiempo", ax=ax)
    plt.tight_layout()
    plt.show()
    ```

- **Q17 – Perfiles NLP sospechosos por persona** (`question17_nlp_person_profiles`):
  - **Metodología:** combina `reports["persona"][timeframe]` y `reports["persona_concepto"][timeframe]` para cuantificar transacciones NLP sospechosas, conceptos únicos y riesgo promedio por persona. Calcula proporciones respecto al total de movimientos, integra los principales conceptos detectados y evalúa el flujo neto para contextualizar el comportamiento descrito.
  - **Parámetros clave:** `timeframe`; `top_n` (máximo de personas priorizadas, 15 por defecto).
  - **Ejemplo de uso:**
    ```python
    from experiment_questions import question17_nlp_person_profiles

    q17 = question17_nlp_person_profiles(reports, timeframe="todo_el_tiempo", top_n=5)
    print(q17[["persona", "tx_sospechosas_nlp", "top_conceptos_display"]].head())
    ```
  - **Visualización rápida:**
    ```python
    import matplotlib.pyplot as plt
    from experiment_visualizations import plot_q17_nlp_person_profiles

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_q17_nlp_person_profiles(reports, timeframe="todo_el_tiempo", ax=ax)
    plt.tight_layout()
    plt.show()
    ```

## CLI
```bash
python -m coi_fraud --csv ./mis_transacciones.csv --out ./forensic_outputs
```
