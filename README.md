# Inferencia Text-to-SQL en el servidor

El script genera consultas a partir de la pregunta y del esquema completo, como en el escenario A de la propuesta. Usa Spider `validation` (1.034 preguntas), BIRD `validation` (1.534) y Catastro `test_transfer` (80). Los pesos se cargan de uno en uno en BF16 y la generación usa batch de uno, sin muestreo, con hasta 512 tokens de salida. BIRD usa la pregunta y el esquema por defecto. Agrega `--use-evidence` para incluir su evidencia original.

## Modelos y memoria

| Opción de `--model` | Checkpoint | Contexto del checkpoint usado |
|---|---|---:|
| `qwen-coder` | [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | 32.768 |
| `qwen-instruct` | [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 32.768 |
| `aya-expanse` | [Aya Expanse 8B](https://huggingface.co/CohereLabs/aya-expanse-8b) | 8.192 |
| `llama-instruct` | [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 131.072 |

BF16 requiere aproximadamente 15–17 GB para los pesos de cada uno de estos modelos, más memoria temporal y caché de atención. En un GB10 con 64 o 128 GB de memoria unificada debería ser viable cargarlos uno por uno. La salida de `nvidia-smi` que compartiste confirma el GB10 y CUDA 13.0. Comprueba la memoria disponible con `free -h`, el espacio de caché con `df -h "$HOME"` y el funcionamiento de CUDA con el comando de abajo. Para almacenar los cuatro checkpoints, reserva del orden de 70–80 GB libres. [Especificaciones de NVIDIA](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) · [Memoria unificada y nvidia-smi](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)

El límite práctico inicial es 32.768 tokens y se aplica el menor entre ese valor y el contexto del checkpoint. La reserva de salida cuenta dentro de ese límite. El script cuenta tokens con el tokenizer de cada modelo y conserva el esquema completo. Catastro tiene un esquema demasiado largo para Aya Expanse 8B. Sus ejemplos que excedan el límite se guardarán con `excluded_context_length`, con el conteo real de su tokenizer. El parámetro `--context-limit` cambia el límite práctico dentro del contexto que admite cada modelo.

## Copiar y preparar

Desde PowerShell, en la raíz del repositorio local, copia esta carpeta completa. Incluye los módulos compartidos en `src/`. Los dos archivos de datos que ya están en el servidor se leen desde `~/text-to-sql/`.

```powershell
scp -r .\scripts_servidor eguards@gx10-d08b:~/text-to-sql/
```

En la terminal del servidor, activa el entorno y comprueba la GPU. La prueba ejecuta una operación de atención real en BF16 y solo requiere PyTorch.

```bash
cd ~/text-to-sql
conda activate text2sql
python scripts_servidor/inferir.py --check-gpu
```

Si el entorno carece de PyTorch con CUDA, instala la compilación de CUDA 13.0 en ese mismo entorno y repite `--check-gpu`. [Guía oficial de NVIDIA para instalar PyTorch CUDA 13 en DGX Spark](https://build.nvidia.com/spark/llama-factory/instructions) · [Ruedas oficiales de PyTorch para Linux ARM64](https://download.pytorch.org/whl/cu130/torch/)

```bash
python -m pip install --upgrade 'torch==2.10.0+cu130' --index-url https://download.pytorch.org/whl/cu130
python scripts_servidor/inferir.py --check-gpu
```

Después instala las dependencias y comprueba que el script encuentre las preguntas y sus esquemas.

```bash
python -m pip install -r scripts_servidor/requirements.txt
python scripts_servidor/inferir.py --plan --model all
```

Para descargar Aya y Llama, pide acceso desde sus fichas de Hugging Face y ejecuta `hf auth login` en el servidor con esa cuenta. Conserva el token en el almacén de credenciales de Hugging Face. [Acceso a modelos restringidos](https://huggingface.co/docs/hub/models-gated)

## Ejecutar

Primera prueba con tres preguntas de Spider `validation`. El primer uso descarga el modelo a la caché de Hugging Face.

```bash
python scripts_servidor/inferir.py --model qwen-coder --dataset spider --limit 3
```

Para ejecutar las tres particiones completas con los cuatro modelos, en secuencia:

```bash
python scripts_servidor/inferir.py --model all
```

Usa una opción de la tabla para correr un solo modelo. `--dataset` admite `spider`, `bird`, `catastro` y `all`. Si los datos están en otro lugar, pasa `--master /ruta/master_dataset.parquet` y `--schemas /ruta/schemas.json`. Con `--output` puedes cambiar la carpeta de resultados.

## Archivos guardados

Cada corrida escribe en `resultados_inferencia/<modelo>/<fuente>_<partición>_<huella>/`.

- `predictions.jsonl` guarda una fila por pregunta procesada, con su `example_id`, su `schema_id`, `sql_generated`, la salida original, los tokens, el tiempo de generación y el estado.
- `manifest.json` registra los ejemplos seleccionados, la revisión del modelo, el contexto, la configuración y las versiones de la corrida.

Si se interrumpe una corrida, repite el comando con los mismos datos, versión del script y entorno. Se conservan las filas ya registradas. Una selección pequeña con `--limit` tiene su propia carpeta. El script lee el maestro y el catálogo en modo de solo lectura. Las salidas son predicciones para revisar o evaluar más adelante.
