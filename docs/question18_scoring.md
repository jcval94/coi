# Metodología de scoring para `question18_user_risk_scores`

Este documento describe la lógica que agrega señales de riesgo por persona en `question18_user_risk_scores`. Todas las casuísticas se agregan a nivel persona para la ventana solicitada y se expresan como puntajes discretos:

* **Alto** → 3
* **Medio** → 2
* **Bajo** → 1 (incluye los casos en los que la persona no participa en la casuística)

Los cortes entre niveles usan percentiles sobre la métrica primaria de cada casuística. El percentil 90 marca el límite de "Alto" cuando existen suficientes datos (>1 fila); el percentil 60 o la mediana se usa como referencia para "Medio". Cuando sólo hay un registro se asigna automáticamente el nivel alto.

## Integración general

1. Cada casuística se normaliza en un `DataFrame` con columnas por persona que contienen:
   * Métricas numéricas (conteos, montos, pesos combinados).
   * Listas resumidas (contrapartes, referencias, conceptos).
   * Texto interpretativo por casuística.
2. Tras normalizar, se ejecuta `_score_metric_series` sobre la métrica principal de la casuística. El helper convierte la serie en numérica, calcula los percentiles relevantes y asigna el score 3/2/1. Los valores ausentes o ≤0 reciben 1.
3. Se almacena para cada casuística el score, el tier (`Alto`, `Medio`, `Bajo`) y un detalle textual breve que se usa tanto en columnas individuales como en la explicación consolidada.
4. Se definen pesos relativos por casuística (ver tabla) y se calcula `casuistica_score_total = Σ (score_i × peso_i)` junto con `casuistica_score_promedio` normalizado por la suma de pesos.
5. El resumen `casuistica_resumen` recoge hasta tres casuísticas con score > 1 ordenadas por score y peso para mostrar en la interpretación principal.
6. La priorización final ordena por `casuistica_score_total`, riesgo promedio, desbalance neto absoluto y banderas.

## Pesos y métricas por casuística

| Casuística | Métrica primaria | Detalles agregados | Peso |
| --- | --- | --- | --- |
| NLP manager-subordinado | `tx_manager_nlp` (conteo total de pagos con etiquetas NLP) | Rol (manager/subordinado), conceptos detectados, monto total | 1.1 |
| Conceptos NLP severos | Máximo `risk_p95` asociado a los conceptos de la persona | Conceptos coincidentes | 0.9 |
| Quid pro quo | `tx_quid_pairs × max(quid_score)` | Contrapartes apareadas | 1.2 |
| Valor vs. carga | Suma de `abs(delta días)` + `feat_quid_score` | Contrapartes y responsables | 1.0 |
| Referencias reutilizadas | Total de transacciones con la referencia | Referencias normalizadas y contrapartes | 1.0 |
| Centralizadores | `inflow` acumulado del receptor | Número de emisores únicos y pagos | 1.1 |
| Desbalance neto | `|desbalance_persona_monto_neto|` | Meses extremos al enviar/recibir | 1.4 |
| Caso 13 (receptores nuevos) | Monto total recibido | Tx y emisores únicos | 1.3 |
| Caso 14 (veteranos desde nuevos) | Monto recibido desde emisores nuevos | Tx y emisores únicos | 1.2 |
| Yo-yo | `tx_yo_yo_totales + racha_max + (riesgo_max - 1)` | Contrapartes en la racha | 1.0 |
| Cercanía a umbral | `tx_near_totales + monto_total_near / 1000` | Contrapartes involucradas | 0.9 |
| Fraccionamiento crónico | `transacciones_fraccionadas + monto_fraccionado_total / 1000` | Contrapartes | 1.4 |
| Préstamos impagos | `monto_prestamos_incumplidos + 1000×prestamos + 500×eventos` | Contrapartes | 1.3 |
| Pagos recurrentes tipo nómina | `monto_total + 1000×meses_recurrentes` | Contrapartes recurrentes | 1.2 |

Las métricas monetarias se expresan como floats y se combinan con conteos para dar más peso a concentraciones relevantes. Cuando una métrica principal es cero o no hay registros, la persona obtiene score 1 (Bajo) en esa casuística.

## Interpretabilidad

* Cada casuística genera un detalle textual que menciona los conteos, montos y contrapartes relevantes.
* El resumen `casuistica_resumen` concatena las tres casuísticas con mayor contribución (score > 1) siguiendo el formato `"<label> <tier>: <detalle>"`.
* En la explicación principal (`interpretabilidad`) se anexa el resumen, junto con el `casuistica_score_total` y el promedio ponderado.

## Nivel de agregación

Todas las casuísticas se consolidan a nivel persona sin distinción temporal dentro de la ventana analizada. Cuando la casuística proviene de pares (emisor ↔ receptor) se replica la contribución para cada persona involucrada antes de agrupar.

