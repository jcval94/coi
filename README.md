# COI/Fraud mensual + NLP MX + Seaborn Viz + Q&A

## Requisitos
- Instalar dependencias con `pip install -r requirements.txt` (incluye pandas, numpy, seaborn, scikit-learn y scipy; estas dos últimas son opcionales si no se entrenan embeddings)

## Instalación en Google Colab
1. Abre un cuaderno nuevo en [Google Colab](https://colab.research.google.com/), ve a **Entorno de ejecución → Cambiar tipo de entorno de ejecución** y confirma que usas Python ≥3.10.
2. Descarga el código directamente desde GitHub en la carpeta de trabajo de Colab (`/content`). El siguiente bloque es reproducible cada vez que necesites partir de cero y agrupa todos los pasos opcionales y automatizados en un único fragmento de referencia:
   ```python
   # Paso 1. Clona o actualiza el repositorio en /content
   %cd /content
   !rm -rf coi  # elimina una copia previa si la hubiera
   !git clone --depth 1 --branch main https://github.com/tu-org/coi.git coi
   %cd /content/coi

   # Paso 2 (opcional). Si subiste un .zip en lugar del repo, descomenta y ajusta la ruta
   # from zipfile import ZipFile
   #
   # with ZipFile("/content/coi_fraud_mensual_viz_qa.zip", "r") as z:
   #     z.extractall("/content")
   # %cd /content/coi

   # Paso 3. Instala las dependencias (idempotente entre sesiones)
   %pip install -q pandas numpy seaborn scikit-learn scipy

   # Paso 4. Añade el repositorio al sys.path para importar coi_fraud desde /content
   import sys

   if "/content" not in sys.path:
       sys.path.append("/content")

   # Paso 5. Valida que todo se importe correctamente
   import coi_fraud
   from coi_fraud import run_pipeline

   # Consejo. Monta Google Drive para trabajar con datasets grandes
   from google.colab import drive

   drive.mount("/content/drive")

   # Flujo automatizado: clona, instala, ejecuta el pipeline y exporta casuísticas de una sola vez
   from colab_usage import run_full_colab_flow

   salidas = run_full_colab_flow(
       csv_input_path="/content/mis_transacciones.csv",  # tu archivo con columnas mínimas
       repo_url="https://github.com/tu-org/coi.git",
       branch="main",
       target_dir="/content/coi",
       output_dir="/content/coi_casuisticas",
       include_empty=False,  # cambia a True si quieres CSV incluso sin hallazgos
   )

   # Muestra rutas generadas por casuística y periodo
   salidas

   # Variante manual: controla cada paso con helpers específicos
   from colab_usage import (
       setup_environment,
       run_pipeline_from_csv,
       export_casuistica_to_csv,
   )

   repo_path = setup_environment(force_refresh=False)
   reports = run_pipeline_from_csv("/content/mis_transacciones.csv", repo_dir=repo_path)
   export_casuistica_to_csv(reports, "/content/coi_casuisticas")
   ```

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
Una vez clonada la repo, basta con cargar tu CSV y ejecutar el pipeline:

```python
import pandas as pd
from coi_fraud import run_pipeline
from coi_fraud.viz import plots
from coi_fraud.analysis import qa

df = pd.read_csv("/content/mis_transacciones.csv")  # columnas mínimas: user_id, receptor-user_id, load_date, movement_amount, transaction_desc
reports = run_pipeline(df)

# Exporta todas las casuísticas de manera explícita
from colab_usage import export_casuistica_to_csv
export_casuistica_to_csv(reports, "/content/coi_casuisticas")

# gráficos y consultas
plots.plot_person_imbalance_bar(reports)
qa.desbalance_personas(reports).head()
```

### 5.1 Describir preguntas e interpretabilidad
Para entender qué cubre cada pregunta Q1–Q18 directamente desde Colab puedes
apoyarte en `colab_usage.question_overview` y en el resumen de
interpretabilidad:

```python
from colab_usage import question_overview, summarize_question_interpretability

descripcion_preguntas = question_overview()
descripcion_preguntas

interpretabilidad = summarize_question_interpretabilidad(reports, timeframe="todo_el_tiempo")
interpretabilidad[["question_id", "filas", "interpretabilidad_ejemplos"]].head()
```

### 5. Generar un dataset de prueba diverso
```python
from coi_fraud import generate_diverse_dataset

dataset = generate_diverse_dataset()  # 6 000 filas por defecto
```

## Preguntas de experimentación (Q1–Q7)
El módulo `experiment_questions.py` genera respuestas tabulares para siete preguntas recurrentes a partir del diccionario de `reports` producido por `run_pipeline`. Todas las funciones aceptan `timeframe` (por defecto `"todo_el_tiempo"`) y devuelven columnas de interpretabilidad en español listando la lógica aplicada.

- **Q1 – Manager con conceptos NLP sospechosos** (`question1_manager_nlp`):
  - **Metodología:** filtra transacciones manager-subordinado en `reports["transaccion"][timeframe]`, concatena campos `nlp_concepto_sospechoso`, `descripcion` y `tx_tags`, y ejecuta coincidencias por expresiones regulares contra las categorías `("SOBORNO", "FACILITACIÓN", "OFUSCACIÓN", "EXTORSIÓN", "FAVORES SEXUALES")` y sus sinónimos (`NLP_CATEGORY_SYNONYMS`). Agrupa por mes y par jerárquico, sumando `tx_count`, `monto_total` y recopilando en una lista todos los conceptos detectados para construir textos explicativos.
  - **Parámetros clave:** `timeframe`; `categories` (lista de categorías NLP, por defecto las cinco anteriores); `direction` (``"manager_a_subordinado"`` por defecto, o ``"subordinado_a_manager"`` para invertir el flujo).
  - **Visualización rápida:**
    ```python
    import matplotlib.pyplot as plt
    from experiment_visualizations import plot_q1_manager_nlp

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_q1_manager_nlp(reports, timeframe="todo_el_tiempo", ax=ax)
    plt.tight_layout()
    plt.show()
    ```
    Para invertir rápidamente la dirección mostrada basta con indicar
    `invert_direction=True` al invocar la función de visualización.

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
  - **Campos clave para interpretar el resultado:**
    - `quid_tx_count`: número de transacciones detectadas para el par emisor↔receptor que superaron el umbral mínimo de `feat_quid_score`. Se obtiene al contar las filas agrupadas por `feat_quid_pair_key`. Un conteo alto indica recurrencia en el posible intercambio de favores y, por lo tanto, mayor riesgo de conflicto de interés sostenido en el tiempo.
    - `quid_score_max`: puntaje máximo de quid pro quo alcanzado por el par. Se calcula como el valor máximo de `feat_quid_score` dentro del grupo. Un máximo elevado significa que, al menos en una ocasión, la transacción mostró señales muy fuertes de intercambio indebido, lo que amerita revisión prioritaria.
    - `quid_score_avg`: promedio de los puntajes `feat_quid_score` de ese par. Resume la intensidad típica del patrón sospechoso; si el promedio es alto, no se trata de un evento aislado sino de una dinámica repetida, aumentando la probabilidad de conflicto de interés.
    - `quid_manager_ratio`: proporción de transacciones del par en las que la relación detectada involucra a un manager (derivada de `feat_quid_rel_label`). Se calcula dividiendo el número de eventos con etiqueta de jefatura entre `quid_tx_count`. Cuanto más se acerque a 1, mayor es la intervención de figuras con poder de decisión, lo que incrementa el riesgo de que exista influencia indebida.
    - `quid_aprob_ratio`: porcentaje de transacciones del par que contaron con una aprobación identificada (`feat_quid_has_approval`). Se obtiene como la media de ese indicador booleano. Un valor alto sugiere que las operaciones sospechosas reciben validaciones formales, lo que puede ocultar conflictos de interés institucionalizados.
    - `quid_comp_ratio`: proporción de eventos donde se detectó algún elemento de compensación (`feat_quid_has_comp`). También se calcula como la media del indicador correspondiente. Si el ratio es elevado, hay evidencia de que el beneficio no solo fue autorizado sino que vino acompañado de contraprestaciones, fortaleciendo la hipótesis de quid pro quo.
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
  - **Metodología:** resume `reports["transaccion"][timeframe]` por mes y receptor (`receptor-user_id`) y calcula las siguientes métricas antes de ordenar de mayor a menor por mes y redactar la interpretabilidad:
    - `inflow`: suma de `movement_amount` (columna `COL_AMOUNT`) recibida por el receptor durante el mes.
    - `emisores_unicos`: conteo de emisores distintos (`user_id`) que enviaron fondos a ese receptor en el mes (`nunique`).
    - `n_tx`: número total de transacciones recibidas (`count` sobre `movement_amount`).
    - `centralidad`: producto `inflow * emisores_unicos`, utilizado como métrica de priorización.
    - Además, se calcula `risk_avg` como el promedio de `risk_score` asociado a las transacciones del receptor.
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
  - **Metodología:** localiza pares que envían y reciben dinero entre sí en rápida sucesión. Primero utiliza la bandera
    `sig_yoyo` dentro de `reports["transaccion"][timeframe]` para detectar secuencias ida-vuelta; cuando no existe esa bandera,
    recalcula la racha comparando cada transacción con su reversa dentro de ventanas móviles (8, 24 y 72 horas) o, en último
    caso, marcando pares que operan en ambas direcciones. Luego cruza con `reports["par_personas"]` para anexar el riesgo
    histórico del par y prioriza los resultados que superan un mínimo de eventos consecutivos y un umbral de riesgo.
  - **Parámetros clave:** `timeframe`; `min_consecutive` (mínimo de eventos consecutivos, por defecto `2`); `risk_threshold`
    (riesgo mínimo del par, por defecto `1.8`).
  - **Ejemplo sencillo:**
    ```python
    import pandas as pd
    from experiment_questions import question10_yoyo_streaks

    # Creamos un subconjunto mínimo que imita el formato de reports
    tx = pd.DataFrame(
        {
            "sender_id": ["A", "B", "A", "B"],
            "receiver_id": ["B", "A", "B", "A"],
            "fecha_hora_ts": pd.to_datetime(
                ["2024-01-01 10:00", "2024-01-01 11:00", "2024-01-02 09:00", "2024-01-02 10:00"]
            ),
            "month_id": ["2024-01"] * 4,
            "sig_yoyo": [True, True, True, True],
            "risk_score": [2.1, 2.3, 2.2, 2.4],
        }
    )
    reports = {"transaccion": {"todo_el_tiempo": tx}, "par_personas": {"todo_el_tiempo": pd.DataFrame()}}

    q10 = question10_yoyo_streaks(reports, timeframe="todo_el_tiempo", min_consecutive=2)
    print(q10[["par_bidir", "racha_max_yo_yo", "tx_yo_yo_totales"]])
    # Resultado: una fila para el par "A⇄B" con racha máxima de 4 y 4 transacciones yo-yo
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
  - **Metodología:** procesa `reports["transaccion"][timeframe]` para identificar pares con bandera `sig_near_thr` y deltas
    pequeños (`feat_delta_near_thr`) respecto a umbrales regulatorios. Si la métrica falta, calcula automáticamente la distancia
    al umbral más cercano dentro del conjunto validado de montos relevantes: 500, 750, 1 000, 1 500, 2 000, 3 000, 5 000, 7 500,
    10 000, 15 000 y 20 000 unidades monetarias. El algoritmo verifica que las transacciones queden dentro del `delta_limit`
    configurado y consolida meses con recurrencia, montos totales y riesgo máximo del par.
  - **Parámetros clave:** `timeframe`; `min_months` (meses mínimos con recurrencia, por defecto `3`); `delta_limit` (diferencia
    máxima al umbral, por defecto `10.0`).
  - **Validación de umbrales:**
    ```python
    from experiment_questions import question11_near_threshold_structuring

    q11 = question11_near_threshold_structuring(reports, timeframe="todo_el_tiempo", min_months=1)
    assert (q11["delta_promedio"] <= 10.0).all(), "Existen pares con delta_promedio fuera del umbral validado"
    print("Umbrales regulatorios verificados: las transacciones cercanas están dentro de ±10 unidades del umbral más próximo.")
    ```
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

- **Q18 – Personas con riesgo agregado y banderas** (`question18_user_risk_scores`):
  - **Metodología:** utiliza `reports["persona"][timeframe]` para combinar el riesgo promedio (`risk_avg_person`) con el flujo neto (`sum_emit - sum_recv`), el desbalance mensual (`desbalance_persona_*`) y las tasas de señales por persona. Calcula un ranking priorizando riesgo, magnitud del desbalance y banderas activas, además de sintetizar las tres señales más destacadas por frecuencia relativa.
  - **Parámetros clave:** `timeframe`; `top_n` (máximo de personas en el ranking, 25 por defecto).
  - **Salida en CLI:** el resumen estándar ahora lista hasta 10 filas (y sus interpretabilidades) para Q18, incluso si las personas adicionales tienen riesgo promedio 0.
  - **Ejemplo de uso:**
    ```python
    from experiment_questions import question18_user_risk_scores

    q18 = question18_user_risk_scores(reports, timeframe="todo_el_tiempo", top_n=10)
    print(q18[["persona", "risk_avg_person", "banderas_destacadas"]].head())
    ```
  - **Visualización rápida:**
    ```python
    import matplotlib.pyplot as plt

    q18_chart = q18.sort_values("risk_avg_person", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(q18_chart["persona"], q18_chart["risk_avg_person"], color="#d95f02")
    ax.set_ylabel("Riesgo promedio")
    ax.set_xlabel("Persona")
    ax.set_title("Q18 – Ranking de riesgo promedio por persona")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    ```

## CLI
```bash
python -m coi_fraud --csv ./mis_transacciones.csv --out ./forensic_outputs
```

## Glosario de señales y métricas clave

| Término | Definición resumida | Cómo se calcula/activa |
| --- | --- | --- |
| **Quid Pro Quo** (`sig_quid_pro_quo`, `feat_quid_score`, `feat_quid_value_vs_load_days`, `feat_quid_rel_label`) | Señal que busca intercambios jerárquicos con aprobaciones y compensaciones potencialmente recíprocas en ventanas cortas. | Combina relaciones manager-subordinado, coincidencias textuales de aprobaciones/compensaciones, montos atípicos por emisor, cercanía a umbrales y desfases carga vs. valor para generar un puntaje; al superar el umbral configura la bandera y conserva metadatos normalizados por par.【F:coi_fraud/features/quid.py†L176-L287】 |
| **Yo-Yo** (`sig_yoyo`) | Marca pares que envían y reciben montos similares en rachas de ida y vuelta. | Agrupa pares bidireccionales, ordena por tiempo y activa la bandera cuando encuentra transferencias opuestas dentro de una ventana de horas y con diferencias menores a una tolerancia porcentual.【F:coi_fraud/features/yoyo.py†L7-L43】 |
| **Smurfing** (`sig_smurf`) | Identifica depósitos fragmentados en montos pequeños que, sumados, superan umbrales dentro de un periodo corto. | Ordena las transacciones por par, calcula ventanas deslizantes de días y enciende la señal si el acumulado rebasa los umbrales configurados sin que algún monto individual los exceda.【F:coi_fraud/features/smurf.py†L7-L48】 |
| **Change Point / Nuevo enlace** (`sig_pair_change_point`, `sig_pair_new_edge`, `feat_pair_month_amount_ratio`) | Detecta saltos abruptos o la aparición repentina de relaciones entre pares. | Resume montos y conteos mensuales por par, compara contra el mes anterior y marca picos grandes o inicios con suficiente separación temporal para determinar cambios estructurales.【F:coi_fraud/features/change_points.py†L10-L140】 |
| **NLP corporativo** (`nlp_concepto_sospechoso`, `feat_nlp_risk_points`, `feat_nlp_coi_score`) | Clasificación automática de descripciones que sugiere conceptos sospechosos o eventos corporativos relevantes. | Ejecuta el modelo `nlp_mx_etiquetar_transacciones_pro` sobre la descripción y la relación declarada, devolviendo conceptos, puntajes de riesgo y atributos lingüísticos asociados a cada transacción.【F:coi_fraud/features/nlp_mx.py†L7-L22】 |
| **Desbalanceo por persona** (`desbalance_persona_monto_neto`, `desbalance_persona_meses_*`) | Mide diferencias persistentes entre lo emitido y recibido por cada persona. | Calcula el neto emitido vs. recibido, razones, z-scores y meses con extremos estadísticos para cuantificar la magnitud y recurrencia del desbalance.【F:coi_fraud/aggregate/persons.py†L491-L595】 |
| **Reutilización de referencias** (`feat_reference_norm`, `sig_reference_reuse`) | Señala referencias de pago reutilizadas por múltiples pares en poco tiempo. | Normaliza textos de referencia, filtra longitudes mínimas y activa la señal cuando un identificador aparece en al menos dos pares dentro de la ventana de días configurada.【F:coi_fraud/features/reference_reuse.py†L31-L78】 |
| **Cercanía a umbrales** (`sig_near_thr`, `feat_delta_near_thr`) | Destaca montos que caen muy cerca de límites regulatorios predefinidos. | Calcula la distancia mínima a la lista de umbrales y marca la transacción si el valor cae dentro del delta permitido, almacenando la diferencia absoluta para referencia.【F:coi_fraud/features/near_threshold.py†L4-L17】 |
