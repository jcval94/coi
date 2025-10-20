# COI/Fraud mensual + NLP MX + Seaborn Viz + Q&A

## ¿Qué problema resuelve este proyecto?
Imagina que una empresa necesita vigilar que sus empleados no usen su cargo para beneficiarse injustamente (lo que conocemos como *conflicto de interés*). Este repositorio reúne un conjunto de herramientas que:

- Analizan historiales de transacciones para encontrar patrones sospechosos (por ejemplo, intercambios de favores, montos repetidos o referencias de pago clonadas).
- Usan modelos de lenguaje para detectar descripciones extrañas en pagos y así levantar banderas de alerta.
- Generan reportes, gráficos y respuestas guiadas que ayudan a los equipos de auditoría a explicar qué ocurre con cada persona, par o transacción.

La idea es que, aun si apenas estás aprendiendo sobre datos, puedas seguir los pasos, cargar un archivo CSV y obtener pistas claras sobre posibles conflictos de interés dentro de una organización.

## Requisitos
- Instalar dependencias con `pip install -r requirements.txt` (incluye pandas, numpy, seaborn, scikit-learn y scipy; estas dos últimas son opcionales si no se entrenan embeddings)

## Instalación como paquete
Si clonas el repositorio y quieres reutilizarlo como dependencia en otros proyectos, puedes instalarlo directamente con pip:

```bash
pip install .
```

Para un modo editable (útil durante el desarrollo) utiliza:

```bash
pip install -e .
```

> 💡 En entornos corporativos sin acceso saliente a PyPI asegúrate de tener
> preinstalados los paquetes de `requirements.txt` y `setuptools` en tu
> ambiente virtual. Si necesitas evitar el aislamiento de construcción, puedes
> ejecutar `pip install --no-build-isolation -e .` para reutilizar las
> dependencias ya presentes.

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
  - **¿Qué detecta?** Relaciones jefe⇄subordinado donde las descripciones de los pagos contienen palabras asociadas a sobornos, favores indebidos o presiones.
  - **¿Cómo lo hace?** Junta los textos relevantes de cada transacción y los compara con un listado de categorías sospechosas. Después resume los montos y la cantidad de pagos por mes para contar la historia en lenguaje sencillo.
  - **Parámetros clave:** `timeframe`; `categories` (lista de conceptos a vigilar, ya configurada con las cinco categorías principales); `direction` (para analizar de jefe a subordinado o al revés).
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
  - **¿Qué detecta?** Cuáles son los conceptos sospechosos que se repiten con montos altos o riesgosos a lo largo del tiempo.
  - **¿Cómo lo hace?** Reutiliza las coincidencias de Q1, agrupa por categoría y mes, y calcula cuántas transacciones hubo y qué tan altos fueron sus puntajes de riesgo.
  - **Parámetros clave:** `timeframe`; las mismas categorías internas de NLP que usa Q1.
  - **Visualización rápida:**
    ```python
    import matplotlib.pyplot as plt
    from experiment_visualizations import plot_q2_manager_concepts

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_q2_manager_concepts(reports, timeframe="todo_el_tiempo", ax=ax)
    plt.tight_layout()
    plt.show()
    ```

- **Q3 – Pares con rasgos Algo por Algo** (`question3_quid_pairs`):
  - **¿Qué detecta?** Duplas de personas que parecen intercambiar favores entre sí (el “yo te ayudo si tú me ayudas”), mostrando en lenguaje sencillo cuántas veces sucede, qué tan grave es y cuál es la evidencia más clara.
  - **¿Cómo lo hace?** Revisa resúmenes y transacciones detalladas, prioriza los pares con puntajes altos de “algo por algo”, participación de jefes y menciones de aprobaciones o compensaciones. Si no hay suficientes hallazgos, relaja los filtros para no perder pistas y conecta las transacciones originales para describir fechas, montos, jerarquías, desfases y textos.
  - **Cómo leer sus columnas principales:**
    - `cantidad_movimientos_con_indicio_de_algo_por_algo`, `puntaje_algo_por_algo_mas_alto_en_el_par`, `puntaje_algo_por_algo_promedio_en_el_par` y `porcentaje_movimientos_donde_participa_un_jefe`: muestran cuántas veces se detectó el patrón, qué tan fuerte fue y si hubo mandos involucrados.
    - `porcentaje_movimientos_con_texto_de_aprobacion` / `porcentaje_movimientos_con_texto_de_compensacion` más `cantidad_movimientos_con_texto_de_aprobacion` / `cantidad_movimientos_con_texto_de_compensacion`: indican con qué frecuencia aparecen palabras de aprobación o de promesa de pago y en cuántos movimientos específicos.
    - `monto_total_de_los_movimientos_relacionados`, `monto_mas_alto_de_los_movimientos_relacionados`, `riesgo_maximo_de_los_movimientos_relacionados`, `riesgo_promedio_de_los_movimientos_relacionados`: dimensionan la parte económica y el riesgo observado.
    - `fecha_del_primer_movimiento_relacionado`, `fecha_del_ultimo_movimiento_relacionado`, `tipos_de_relacion_observados_entre_las_personas`: explican la ventana temporal y los lazos jerárquicos o familiares mencionados.
    - Los campos `ejemplo_clave_*` describen el movimiento más ilustrativo (fecha, monto, puntaje, riesgo, relación, desfase, si menciona aprobación o compensación, descripción, referencia y si apareció tras relajar filtros).
    - `interpretabilidad`: texto simple que resume el caso y aclara si se usaron filtros relajados.
  - **Ejemplo con “palitos y bolitas”:**
    1. Supón que “Jefa_Luisa” autoriza tres gastos a “Proveedor_Julio”. Eso llena `cantidad_movimientos_con_indicio_de_algo_por_algo` con `3` y el identificador queda `Jefa_Luisa->Proveedor_Julio`.
    2. Como dos mensajes dicen “aprobado por dirección”, `porcentaje_movimientos_con_texto_de_aprobacion` muestra `67%` y `cantidad_movimientos_con_texto_de_aprobacion` vale `2`.
    3. El pago más alto fue de 9,000, por eso `monto_mas_alto_de_los_movimientos_relacionados` enseña `9,000.00` mientras que el total suma `18,500.00`.
    4. Si el modelo calculó puntajes de 3.8, 3.2 y 2.9, entonces `puntaje_algo_por_algo_mas_alto_en_el_par` vale `3.8` y el promedio `3.3`.
    5. El bloque `ejemplo_clave_*` copia el movimiento más claro (por ejemplo el de 9,000) para que puedas leer la fecha exacta, la relación “Jefa → Proveedor”, el desfase de días y el texto libre, todo en una sola línea.
    6. Finalmente `interpretabilidad` junta esas piezas y explica en un párrafo por qué ese par luce riesgoso.
  - **Parámetros clave:** `timeframe`; `min_score` (2.2 por defecto); `min_manager_ratio` (0.5 por defecto).
  - **Visualización rápida:**
    ```python
    import matplotlib.pyplot as plt
    from experiment_visualizations import plot_q3_quid_pairs

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_q3_quid_pairs(reports, timeframe="todo_el_tiempo", ax=ax)
    plt.tight_layout()
    plt.show()
    ```
  - **Visualización caso a caso:**
    ```python
    import matplotlib.pyplot as plt
    from experiment_questions import question3_quid_pairs
    from experiment_visualizations import plot_q3_algo_pair_detalle

    par = question3_quid_pairs(reports).query("nivel_respuesta == 'resumen_del_par_algo_por_algo'").iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_q3_algo_pair_detalle(par, ax=ax)
    plt.tight_layout()
    plt.show()
    ```

- **Q4 – Autorizaciones con valor negativo vs. carga** (`question4_quid_negative_value_vs_load`):
  - **¿Qué detecta?** Casos en los que alguien aprueba algo de valor menor al que declaró al inicio (por ejemplo, autoriza un gasto que luego se compensa con un favor).
  - **¿Cómo lo hace?** Busca transacciones con desfases fuertes entre el momento de la carga y el valor final. Si no hay suficientes ejemplos, toma los 10 desfases más pequeños o los puntajes más altos. También identifica quién pudo ser el responsable dentro de la cadena de mando.
  - **Cómo leer los números:** reutiliza las mismas columnas explicadas en Q3 (`cantidad_movimientos_con_indicio_de_algo_por_algo`, `puntaje_algo_por_algo_mas_alto_en_el_par`, `puntaje_algo_por_algo_promedio_en_el_par`, `porcentaje_movimientos_donde_participa_un_jefe`, `porcentaje_movimientos_con_texto_de_aprobacion`, `porcentaje_movimientos_con_texto_de_compensacion`) para dimensionar la recurrencia y la gravedad del hallazgo.
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
  - **¿Qué detecta?** Referencias o textos de pago copiados entre diferentes pares en lapsos cortos.
  - **¿Cómo lo hace?** Revisa resúmenes de referencias y, si es necesario, reconstruye la información normalizando las descripciones. Prioriza los casos donde la misma referencia aparece en varios pares o dentro de un rango de días reducido.
  - **Parámetros clave:** `timeframe` (los demás filtros ya vienen configurados).
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
  - **¿Qué detecta?** Personas que reciben dinero de muchos emisores y concentran montos altos en un mismo mes.
  - **¿Cómo lo hace?** Suma cuánto reciben, cuenta cuántos emisores distintos participan y estima cuántas transacciones llegan. Multiplica el monto por la cantidad de emisores para priorizar a quienes parecen actuar como “imanes” de dinero y añade el riesgo promedio.
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
  - **¿Qué detecta?** Individuos que envían mucho más de lo que reciben (o viceversa) de forma sostenida.
  - **¿Cómo lo hace?** Usa el resumen por persona, ordena por el desbalance absoluto y señala los meses en los que la persona tuvo comportamientos extremos.
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
  - **¿Qué detecta?** Personas recién incorporadas que, en sus primeros meses, reciben montos muy altos.
  - **¿Cómo lo hace?** Usa la bandera `caso13_persona_flag_nuevo_receptor_altos_montos`, se queda con quienes están en el percentil 90 de montos dentro de los primeros seis meses y resume totales, emisores únicos y promedios.
  - **Parámetros clave:** `timeframe`, `new_definition` (por defecto `("months", 6.0)` equivalente al umbral original de 0.5 años).
  - **Ejemplo de uso:**
    ```python
    from experiment_questions import question8_case13_new_employees

    q8 = question8_case13_new_employees(reports, timeframe="todo_el_tiempo")
    print(q8.head())
    ```
  - **Cómo ajustar el criterio de antigüedad:**
    ```python
    # Considera nuevos a quienes lleven ≤3 meses en la organización
    q8_ajustado = question8_case13_new_employees(
        reports,
        timeframe="todo_el_tiempo",
        new_definition=("months", 3.0),
    )

    # O calcula el umbral según el percentil 25 de antigüedad disponible
    q8_percentil = question8_case13_new_employees(
        reports,
        new_definition=("percentile", 0.25),
    )
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
  - **¿Qué detecta?** Personas con mucha antigüedad que empiezan a recibir dinero de emisores recién llegados.
  - **¿Cómo lo hace?** Busca la bandera `caso14_persona_flag_antiguo_recibe_de_nuevos` y, si no existe, reconstruye el cálculo a partir de las transacciones para estimar antigüedad y montos recibidos. Luego resume emisores únicos y promedios para dimensionar la relación.
  - **Parámetros clave:** `timeframe`, `newcomer_definition` (por defecto `("months", 6.0)`) y `veteran_definition` (por defecto `("months", 60.0)` que conserva el umbral de 5 años).
  - **Ejemplo de uso:**
    ```python
    from experiment_questions import question9_case14_veterans_from_newcomers

    q9 = question9_case14_veterans_from_newcomers(reports, timeframe="todo_el_tiempo")
    print(q9.head())
    ```
  - **Cómo ajustar los umbrales:**
    ```python
    q9_personalizado = question9_case14_veterans_from_newcomers(
        reports,
        timeframe="todo_el_tiempo",
        newcomer_definition=("months", 4.0),  # emisores con ≤4 meses
        veteran_definition=("percentile", 0.8),  # imprime la conversión a meses
    )
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
  - **¿Qué detecta?** Pares de personas que se envían dinero ida y vuelta en cuestión de horas o días, como si se estuvieran pasando la misma “canasta” una y otra vez.
  - **Ejemplo corto:** Ana envía 10 peras a Bruno y minutos después Bruno le regresa 10 manzanas. Si repiten el intercambio varias veces, parece actividad normal, pero en realidad es el mismo dinero rotando.
  - **¿Cómo lo hace?** Busca primero la bandera `sig_yoyo`. Si no existe, compara cada transacción con su reversa en ventanas de 8, 24 y 72 horas o, como último recurso, verifica que ambas direcciones estén activas. Después cruza con el historial de riesgo del par para priorizar las rachas largas y con puntajes altos.
  - **Columnas clave para interpretar:** `timeframe` (ventana analizada), `par_bidir` (quiénes participan), `racha_max_yo_yo` (cuántas veces seguidas se repite), `tx_yo_yo_totales` y `meses_con_yo_yo` (frecuencia), `riesgo_max_par` / `riesgo_promedio_par` (historial del par), `riesgo_max_yo_yo` / `riesgo_promedio_yo_yo` (riesgo solo de la racha), `monto_total_yo_yo` / `monto_promedio_yo_yo` (tamaño del flujo) e `interpretabilidad` (resumen en español que explica todo).
  - **Parámetros clave:** `timeframe`; `min_consecutive` (mínimo de eventos seguidos, 2 por defecto); `risk_threshold` (riesgo mínimo del par, 1.8 por defecto).
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
  - **¿Qué detecta?** Pares que mueven dinero muy cerca de límites regulatorios (500, 750, 1 000, 1 500, 2 000, 3 000, 5 000, 7 500, 10 000, 15 000 o 20 000 unidades) como si intentaran esquivar la supervisión sin pasarse del tope.
  - **¿Cómo lo hace?** Busca la bandera `sig_near_thr` y el delta `feat_delta_near_thr`. Si faltan, calcula qué tan lejos está cada transacción del umbral más cercano y se queda con las que caen dentro del `delta_limit` (±10 por defecto). Luego agrupa por mes para ver recurrencia, montos acumulados y riesgo máximo del par.
  - **Parámetros clave:** `timeframe`; `min_months` (meses mínimos con hallazgos, 3 por defecto); `delta_limit` (diferencia máxima permitida respecto al umbral, 10.0 por defecto).
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
- **Q12 – Fraccionamiento crónico** (`question12_smurfing_chronic`):
  - **¿Qué detecta?** Pares que dividen depósitos grandes en muchas transferencias pequeñas (por ejemplo, cinco pagos de 2 000 $) para evitar que un control detecte de golpe los 10 000 $ o 20 000 $ que realmente están moviendo.
  - **Analogía rápida:** Es como si alguien quisiera pasar un costal de arena por una báscula con límite de 10 kg. En vez de cargar un solo costal de 30 kg (que dispararía la alarma), reparte la arena en tres costales de 10 kg y los pasa uno a la vez.
  - **¿Cómo lo hace el código?** El detector de fraccionamiento (`SmurfingDetector`) ordena las transacciones por par emisor→receptor y, dentro de una ventana móvil de 7 días (`smurf_window_days`), suma los montos. Si la suma alcanza 10 000 $ o 20 000 $ (`smurf_thresholds = [10 000, 20 000]`) sin que ninguna transacción individual cruce ese umbral, marca `sig_smurf = True`. Cuando la bandera no existe, la función recalcula la alerta usando cuantiles (umbral 25 %) para identificar montos pequeños repetidos y reconstruir la señal.
  - **Parámetros clave:** `timeframe`; `min_months` (opcional). Cuando se omite, el análisis prioriza el monto fraccionado total sin exigir recurrencia mensual. Si necesitas filtrar por persistencia, ajusta `min_months` a la cantidad deseada.
  - **Columnas de salida relevantes:** `meses_con_fraccionamiento` (en cuántos meses distintos apareció la señal), `transacciones_fraccionadas` (cuántas operaciones pequeñas participaron), `monto_fraccionado_total` (suma de montos), `riesgo_promedio` y `riesgo_maximo` (niveles de riesgo), además de `tendencia_riesgo` para resumir si la alerta va al alza, a la baja o se mantiene estable.
  - **Ejemplo de uso:**
    ```python
    from experiment_questions import question12_smurfing_chronic

    q12 = question12_smurfing_chronic(reports, timeframe="todo_el_tiempo")
    print(q12[["pair", "meses_con_fraccionamiento", "monto_fraccionado_total"]].head())
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
  - **¿Qué detecta?** Pares donde existen préstamos que casi no se pagan (≤50 %) y, al mismo tiempo, ráfagas de transacciones muy seguidas.
  - **¿Cómo lo hace?** Busca las banderas `sig_loan_bad_repay` y `sig_freq` por mes; si no están disponibles, estima quién presta, quién devuelve y qué tan seguido ocurren los pagos usando heurísticas en ambos sentidos. Luego cuenta los meses donde coinciden ambas señales y resume montos y riesgos.
  - **Parámetros clave:** `timeframe`; `min_months` (meses mínimos con coincidencia, 3 por defecto).
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
  - **¿Qué detecta?** Pagos que salen casi siempre en las mismas fechas y montos, imitando una nómina paralela.
  - **¿Cómo lo hace?** Usa la bandera `sig_recurrent` para agrupar pagos que ocurren alrededor del mismo día de corte. Luego busca meses consecutivos con ese patrón y calcula totales, promedios y cantidad de pagos. Si faltan banderas, relaja el criterio para no perder posibles nóminas escondidas.
  - **Parámetros clave:** `timeframe`; `min_months` (meses consecutivos mínimos, 3 por defecto).
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
  - **¿Qué detecta?** Grupos de personas conectadas entre sí que acumulan varias señales (yo-yo, fraccionamiento, quid, etc.) al mismo tiempo.
  - **¿Cómo lo hace?** Lee `reports["clusters_personas"][timeframe]`, suma cuántas señales activas tiene cada cluster, normaliza los montos y ordena los resultados por número de señales, riesgo máximo y volumen. El texto final resalta a la persona más desbalanceada y explica qué porcentaje aporta cada señal.
  - **Parámetros clave:** `timeframe`; `top_n` (número de clusters a mostrar, 10 por defecto).
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
  - **¿Qué detecta?** Operaciones individuales que prenden varias alarmas a la vez (por ejemplo, jerarquía + yo-yo + fraccionamiento).
  - **¿Cómo lo hace?** Revisa `reports["transaccion"][timeframe]` y cuenta cuántas banderas se activan por transacción (jerarquía, yo-yo, fraccionamiento, near-threshold, quid y cambios bruscos). Prioriza las que tienen tres o más señales y, si hay pocos casos, baja el umbral para mostrar ejemplos representativos.
  - **Parámetros clave:** `timeframe`; `top_n` (máximo de transacciones en el listado, 25 por defecto).
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
  - **¿Qué detecta?** Personas que concentran transacciones con conceptos NLP sospechosos y riesgo elevado.
  - **¿Cómo lo hace?** Mezcla la tabla de personas con la de conceptos para contar transacciones sospechosas, conceptos distintos y riesgo promedio por persona. También calcula qué porcentaje representan sobre el total de movimientos y muestra los conceptos más frecuentes junto con el flujo neto para dar contexto.
  - **Parámetros clave:** `timeframe`; `top_n` (máximo de personas en el ranking, 15 por defecto).
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
  - **¿Qué detecta?** El ranking general de personas con mayor riesgo promedio, desbalances fuertes y señales activas.
  - **¿Cómo lo hace?** Usa `reports["persona"][timeframe]` para combinar `risk_avg_person`, el flujo neto (`sum_emit - sum_recv`), los desbalances mensuales (`desbalance_persona_*`) y las tasas de banderas por persona. Con esa mezcla arma un ranking, destaca las tres señales más frecuentes y genera texto explicativo.
  - **Parámetros clave:** `timeframe`; `top_n` (máximo de personas a mostrar, 25 por defecto).
  - **Salida en CLI:** el resumen estándar imprime hasta 10 filas con sus interpretabilidades aunque algunas tengan riesgo promedio 0.
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
| **Algo por Algo** (`sig_quid_pro_quo`, `feat_quid_score`, `feat_quid_value_vs_load_days`, `feat_quid_rel_label`) | Banderas que alertan sobre posibles intercambios de favores entre jefes y subordinados. | Combina relaciones jerárquicas, palabras clave de aprobaciones o compensaciones, montos fuera de lo normal, cercanía a umbrales y desfases entre carga y valor. Con todo eso calcula un puntaje; si rebasa el umbral, activa la señal y guarda los datos por par.【F:coi_fraud/features/quid.py†L176-L287】 |
| **Yo-Yo** (`sig_yoyo`) | Detecta pares que se envían dinero de ida y vuelta con montos casi iguales. | Agrupa el par en ambas direcciones, ordena por tiempo y prende la bandera cuando identifica transferencias opuestas en pocas horas con diferencias mínimas.【F:coi_fraud/features/yoyo.py†L7-L43】 |
| **Fraccionamiento** (`sig_smurf`) | Señala depósitos divididos en partes pequeñas que, sumados, cruzan un umbral. | Ordena las operaciones por par, recorre ventanas de días y enciende la señal si el acumulado supera los límites configurados sin que ningún pago individual lo haga.【F:coi_fraud/features/smurf.py†L7-L48】 |
| **Change Point / Nuevo enlace** (`sig_pair_change_point`, `sig_pair_new_edge`, `feat_pair_month_amount_ratio`) | Advierte cuando surge un par nuevo o cambia drásticamente su nivel de actividad. | Resume montos y conteos mensuales por par, los compara con el mes anterior y marca picos grandes o arranques que aparecen tras un buen tiempo sin relación.【F:coi_fraud/features/change_points.py†L10-L140】 |
| **NLP corporativo** (`nlp_concepto_sospechoso`, `feat_nlp_risk_points`, `feat_nlp_coi_score`) | Clasifica descripciones para encontrar términos sospechosos en lenguaje cotidiano. | Ejecuta el modelo `nlp_mx_etiquetar_transacciones_pro` sobre la descripción y la relación declarada, devolviendo conceptos, puntajes de riesgo y detalles lingüísticos para cada transacción.【F:coi_fraud/features/nlp_mx.py†L7-L22】 |
| **Desbalanceo por persona** (`desbalance_persona_monto_neto`, `desbalance_persona_meses_*`) | Resume quién envía mucho más de lo que recibe (o al revés) de manera sostenida. | Calcula el neto emitido vs. recibido, razones estadísticas y meses extremos para medir la magnitud y la constancia del desbalance.【F:coi_fraud/aggregate/persons.py†L491-L595】 |
| **Reutilización de referencias** (`feat_reference_norm`, `sig_reference_reuse`) | Marca referencias de pago copiadas entre distintos pares en poco tiempo. | Limpia y normaliza los textos, descarta referencias muy cortas y activa la señal cuando el mismo identificador aparece en dos o más pares dentro de la ventana configurada.【F:coi_fraud/features/reference_reuse.py†L31-L78】 |
| **Cercanía a umbrales** (`sig_near_thr`, `feat_delta_near_thr`) | Señala montos que quedan a pocos pesos/unidades de un límite oficial. | Calcula la distancia mínima a la lista de umbrales y activa la bandera si el valor cae dentro del delta permitido, guardando cuánto faltó para llegar al límite.【F:coi_fraud/features/near_threshold.py†L4-L17】 |
