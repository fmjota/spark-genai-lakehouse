# Glosario

## Términos del proyecto

- **NER (Named Entity Recognition)**: reconocimiento de entidades nombradas; aquí, extraer
  fármacos y síntomas de un texto clínico.
- **Análisis de sentimiento**: clasificar un texto como positivo (POS), neutro (NEU) o
  negativo (NEG).
- **Confianza (`confidence`)**: score del modelo en [0, 1] asociado a cada extracción. No es
  necesariamente una probabilidad calibrada (ver ECE).
- **Calibración**: grado en que la confianza reportada coincide con la accuracy real. Un
  modelo al 90% de confianza está calibrado si acierta ~90% de esas veces.
- **ECE (Expected Calibration Error)**: error de calibración esperado; promedio ponderado de
  la brecha entre confianza media y accuracy por tramo. 0 = perfectamente calibrado.
- **Curva de fiabilidad (reliability)**: confianza media vs accuracy observada por tramo de
  confianza; visualiza la calibración.
- **Drift**: desplazamiento de la distribución de los datos entre un período de referencia y
  uno nuevo, que puede degradar el modelo en silencio.
- **PSI (Population Stability Index)**: índice que cuantifica el drift entre dos
  distribuciones. Regla práctica: <0.1 estable, 0.1–0.25 moderado, >0.25 fuerte.
- **KS (Kolmogorov–Smirnov)**: test que contrasta si dos muestras provienen de la misma
  distribución continua (p-valor bajo ⇒ distribuciones distintas).

## Términos técnicos

- **PySpark**: API de Python para Apache Spark, motor de procesamiento distribuido.
- **`mapInPandas`**: operación de Spark que aplica una función sobre `pandas.DataFrame` por
  partición, usando Arrow para el intercambio columnar JVM↔Python.
- **Apache Arrow**: formato de memoria columnar que acelera el traspaso de datos entre la JVM
  y Python (evita serializar fila a fila).
- **UDF (User Defined Function)**: función definida por el usuario que Spark aplica sobre los
  datos. Las **vectorizadas** (pandas UDF / `mapInPandas`) procesan lotes, no filas.
- **Singleton por executor**: patrón en que el modelo se carga una sola vez por proceso
  worker (cache a nivel de módulo), no por fila ni partición.
- **Hugging Face / transformers**: librería de modelos preentrenados de NLP; `pipeline()`
  encapsula tokenizador + modelo para una tarea.
- **Delta Lake**: formato de tabla sobre Parquet con log transaccional; aporta ACID, time
  travel y evolución de esquema.
- **Time travel**: leer una versión pasada de una tabla Delta (`versionAsOf`), para
  reproducir o auditar una corrida anterior.
- **Particionado (`partitionBy`)**: organización física de la tabla por columnas (aquí
  `domain` e `ingest_date`) para consultar sin escanear todo.
- **Pandera**: librería de validación de DataFrames; define un contrato de tipos y reglas.
- **uv**: gestor de entornos/paquetes de Python rápido, con lockfile reproducible.
- **JDK 17/21**: versiones de Java compatibles con Spark 4.x (no soporta Java 25).
- **`GENAI_BACKEND`**: variable de entorno que elige el backend de inferencia
  (`mock` | `hf` | `auto`).
