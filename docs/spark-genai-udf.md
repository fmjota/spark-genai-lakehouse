# El patrón HF-en-Spark sin cuellos de botella

Este es el corazón del proyecto: cómo correr modelos de Hugging Face sobre Spark a gran
escala **sin** que la inferencia se vuelva el cuello de botella. Vale la pena entenderlo a
fondo porque es exactamente lo que se defiende en entrevista.

## El problema

La forma ingenua —una UDF de Python que recibe una fila, carga el modelo y predice— tiene
tres patologías que matan el rendimiento:

1. **Serialización fila a fila** entre la JVM (donde vive Spark) y el proceso Python.
2. **Recarga del modelo** una y otra vez (por fila o por partición).
3. **Sin batching**: el modelo procesa de a un texto, desperdiciando la vectorización.

## La solución, en cuatro técnicas

### 1. `mapInPandas` (Arrow) en vez de UDF fila a fila

`DataFrame.mapInPandas` entrega a la función un **iterador de `pandas.DataFrame`** por
partición, moviendo los datos en bloques **Apache Arrow** (formato columnar). En lugar de
cruzar la frontera JVM↔Python una vez por fila, se cruza una vez por bloque, y el modelo
recibe **lotes** de texto.

```python
# src/genai_etl/models/inference.py
return df.mapInPandas(mapper, schema=output_schema(cfg.model.task))
```

### 2. Singleton del modelo por executor

El modelo se obtiene de un **registry con cache a nivel de módulo**. Como cada proceso
worker de Python importa el módulo una sola vez, los pesos se cargan **una vez por
executor**, no por fila ni por partición:

```python
# src/genai_etl/models/registry.py
_BACKENDS: dict[str, HFBackend] = {}

def get_backend(task, model_name, mode=None):
    key = f"{resolved}:{task}:{model_name}"
    if key not in _BACKENDS:
        _BACKENDS[key] = build_backend(resolved, task, model_name)
    return _BACKENDS[key]
```

Clave: al worker solo viaja el **`model_name` (un string)**, nunca el objeto del modelo. No
se hace `broadcast` de los pesos (pickling de un modelo de torch es costoso y frágil); cada
worker construye/lee su copia desde la cache local de Hugging Face.

### 3. Inferencia por lotes con truncado

Dentro de la partición, la inferencia corre con `batch_size`, `truncation=True` y
`max_length` (parametrizados por dominio en el config). Lotes homogéneos = memoria acotada
y throughput estable.

### 4. Un hilo de torch por worker

Spark ya paraleliza por **particiones → executors**. Si además cada proceso de torch abre N
hilos, hay **sobre-suscripción** de CPU y el rendimiento cae. Por eso, en backends reales:

```python
import torch
torch.set_num_threads(1)   # el paralelismo lo da Spark, no torch
```

## Tabla: cada cuello de botella y su técnica

| Cuello de botella | Síntoma | Técnica | Por qué lo elimina |
|---|---|---|---|
| UDF fila a fila | Serialización JVM↔Python por fila | `mapInPandas` (Arrow) | Transferencia columnar en bloque; el modelo recibe lotes |
| Recarga del modelo | Pesos cargados por fila/partición | Singleton module-level | El proceso worker carga una vez; persiste entre tareas |
| Broadcast del modelo | Pickling caro/frágil de torch | Cargar en el worker (solo viaja el nombre) | No se serializa el objeto modelo |
| Tokenización ineficiente | Padding inconsistente, OOM | `batch_size` + `truncation` + `max_length` | Lotes homogéneos, memoria acotada |
| Sobre-suscripción de CPU | N executors × M hilos saturan | `torch.set_num_threads(1)` | El paralelismo lo da Spark por particiones |

## Por qué escala (el argumento de entrevista)

El paralelismo lo aporta Spark: más datos → más particiones → más executors. Dentro de cada
partición, HF corre en lotes vectorizados con el modelo cargado **una sola vez por proceso**.
El **mismo código** corre en `local[*]` (laptop) y en EMR/Dataproc cambiando solo el master
y el número de workers — no hay reescritura. Escalar es añadir workers.

## El MockBackend (por qué los tests no descargan nada)

`GENAI_BACKEND` selecciona el backend: `mock` (determinista, por reglas de palabras clave,
sin red ni GPU), `hf` (modelos reales) o `auto`. Tests y CI usan `mock`, igual que el
patrón `HEALER_BACKEND` del Proyecto 3: el flujo completo —ingesta, UDF, validación,
calidad, escritura Delta— se ejercita sin descargar un solo modelo, y de forma reproducible.
