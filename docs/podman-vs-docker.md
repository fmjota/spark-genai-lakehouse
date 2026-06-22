# Podman vs Docker: por qué y cómo se ejecuta aquí

Este proyecto se empaqueta con un `Dockerfile` y un `docker-compose.yml` estándar OCI, y
se ejecuta con Podman. A diferencia de la API del proyecto `calibrated-mlops-pipeline`, aquí el contenedor corre un
job batch (spark-submit): se ejecuta, escribe la tabla Delta en el volumen montado y
termina. Por eso necesita Java (Spark 4.x) además de Python, y no expone puertos.

## Qué son

- **Docker.** Usa un daemon central (`dockerd`) que corre como root.
- **Podman.** Motor OCI sin daemon y rootless por defecto; CLI casi idéntica y lee los
  mismos `Dockerfile`/`docker-compose.yml`.

## Diferencias clave

| Aspecto | Docker | Podman |
|---|---|---|
| Arquitectura | Daemon central (`dockerd`) | Sin daemon: cada contenedor es un proceso hijo |
| Privilegios | El daemon corre como root | Rootless por defecto (más seguro) |
| Compose | `docker compose` (plugin v2) | `podman-compose` o `podman compose` |
| Integración Fedora/RHEL | Repo externo | Incluido y soportado |
| Imágenes/Registros | OCI | OCI (mismas imágenes) |

El formato es el mismo (OCI, `Dockerfile`, `compose`), así que el `docker-compose.yml` de
este repo funciona en ambos sin cambios.

## Por qué Podman en este entorno

1. Fedora 44 ya trae Podman; Docker no estaba.
2. **Sin sudo ni daemon root** (rootless): encaja con la regla del proyecto de no tocar el
   sistema sin OK explícito.
3. **Mismo artefacto**: quien prefiera Docker corre el mismo `docker compose up`.

## Cómo se ejecuta

### Con Podman (lo usado aquí)

```bash
# 1. Proveedor de compose a nivel usuario (sin sudo)
uv tool install podman-compose

# 2. Construir la imagen
podman-compose build

# 3. Generar datos de entrada DENTRO del contenedor (quedan en ./data por el volumen)
podman-compose run --rm --entrypoint python pipeline \
    scripts/generate_synthetic.py --domain banking --rows 200 --out data/raw/banking.csv

# 4. Correr el pipeline (escribe la tabla Delta en ./lakehouse)
podman-compose run --rm pipeline --config configs/banking.yaml
```

### Con Docker (si está instalado)

```bash
docker compose build
docker compose run --rm pipeline --config configs/banking.yaml
```

## Detalles del entorno que hubo que resolver

- **Java en la imagen.** Spark 4.x requiere Java 17/21. La base `python:3.12-slim` (Debian
  trixie) no trae Java, así que el `Dockerfile` instala `openjdk-21-jre-headless` y fija
  `JAVA_HOME`.
- **Jars de Delta en el primer arranque.** `delta-spark` descarga sus jars de Maven la primera
  vez que se crea la SparkSession: el contenedor necesita **red** en la primera corrida.
- **SELinux (Fedora Enforcing).** Los volúmenes `./data` y `./lakehouse` llevan la etiqueta
  `z` para relabel (`:z`). Docker la ignora sin problema, así que el archivo sigue portable.
- **`GENAI_BACKEND=mock` por defecto.** El contenedor corre sin descargar modelos; con
  `-e GENAI_BACKEND=hf` usa modelos reales de Hugging Face (descarga la 1ª vez).

## Glosario rápido

- **OCI**: estándar que hace intercambiables a Docker y Podman.
- **Rootless**: contenedores que corren bajo tu usuario, sin root.
- **Daemon**: proceso de fondo siempre activo (Docker lo tiene; Podman no).
- **Relabel SELinux (`z`/`Z`)**: ajusta etiquetas de un volumen para que el contenedor lo lea.
