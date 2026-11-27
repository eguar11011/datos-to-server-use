"""Genera SQL con el esquema completo y guarda las predicciones por modelo."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata as metadata
import os
import platform
import sys
import time
from pathlib import Path

from src.baseline_inference import (canonical_json, clean_sql, digest, messages_for, prediction_records,
                                    read_json, source_hash, write_json)

SCRIPT_DIR = Path(__file__).resolve().parent
PARTITIONS = {"SPIDER": "validation", "BIRD": "validation", "CATASTRO": "test_transfer"}
CHAT_DATE = "26 Nov 2026"
SEED = 42


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("El valor debe ser mayor que cero.")
    return number


def check_gpu():
    import torch

    if not torch.version.cuda or not torch.cuda.is_available():
        raise RuntimeError("No hay CUDA disponible en este PyTorch. Revisa la instalación CUDA 13.")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Se requiere soporte nativo de BF16.")

    with torch.inference_mode():
        sample = torch.ones((1, 1, 16, 64), device="cuda:0", dtype=torch.bfloat16)
        torch.nn.functional.scaled_dot_product_attention(sample, sample, sample)
        torch.cuda.synchronize()

    device = torch.cuda.get_device_properties(0)
    return {
        "architecture": platform.machine(), "python": platform.python_version(),
        "torch": str(torch.__version__), "torch_cuda": torch.version.cuda, "gpu": device.name,
        "compute_capability": f"{device.major}.{device.minor}", "device_total_memory_bytes": device.total_memory,
        "dtype": "bfloat16", "attention": "sdpa",
    }


def select_data(args):
    import pandas as pd

    from src.baseline_data import full_schema, select_questions

    master = pd.read_parquet(args.master)
    catalog = read_json(args.schemas)
    if master["example_id"].isna().any() or master["example_id"].duplicated().any():
        raise ValueError("El maestro debe tener un example_id único por pregunta.")

    sources = PARTITIONS if args.dataset == "ALL" else (args.dataset,)
    jobs = []
    for source in sources:
        split = PARTITIONS[source]
        rows, total = select_questions(master, source, split, args.limit, SEED)
        use_evidence = args.use_evidence and source == "BIRD"
        for row in rows:
            row.pop("sql_gold")
            if not use_evidence:
                row.pop("evidence")
        ids = {row["schema_id"] for row in rows}
        missing = ids - catalog.keys()
        if missing:
            raise ValueError(f"Faltan esquemas en {source}/{split}: {sorted(missing)}")
        schemas = {schema_id: full_schema(catalog[schema_id]) for schema_id in sorted(ids)}
        jobs.append((source, split, rows, total, schemas, use_evidence))
    return jobs


class Generator:
    def __init__(self, spec):
        import torch
        from transformers import AutoConfig, AutoTokenizer

        self.torch = torch
        self.model_id = spec["id"]
        self.revision = spec["revision"]
        config = AutoConfig.from_pretrained(self.model_id, revision=self.revision)
        self.context = min(spec["context"], config.max_position_embeddings)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        self.model = None

    def tokenize(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, truncation=False,
            date_string=CHAT_DATE,
        )

    def load_weights(self):
        from transformers import AutoModelForCausalLM

        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, revision=self.revision, dtype=self.torch.bfloat16,
                device_map={"": 0}, attn_implementation="sdpa", use_safetensors=True,
            ).eval()

    def generate(self, input_ids, max_new_tokens):
        torch = self.torch
        with torch.inference_mode():
            ids = torch.tensor([input_ids], device="cuda:0")
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            output = self.model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=pad_token_id,
            )
            new_tokens = output[0, len(input_ids):].tolist()
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True), len(new_tokens)


def generate_partition(generator, alias, job, args, runtime):
    from filelock import FileLock

    source, split, rows, total, schemas, use_evidence = job
    context = min(args.context_limit, generator.context)
    if args.max_new_tokens >= context:
        raise ValueError(f"La reserva de salida debe ser menor que el contexto de {alias} ({context}).")

    manifest = {
        "model_alias": alias, "model_id": generator.model_id, "revision": generator.revision,
        "source": source, "split": split, "partition_size": total,
        "selected": [{key: row[key] for key in ("example_id", "db_id", "schema_id")} for row in rows],
        "rows_sha256": digest(rows), "schemas_sha256": digest(schemas),
        "script_sha256": source_hash(Path(__file__)),
        "prompt_sha256": source_hash(SCRIPT_DIR / "src" / "baseline_inference.py"),
        "runtime": runtime, "chat_date": CHAT_DATE, "use_evidence": use_evidence,
        "context_limit": context, "max_new_tokens": args.max_new_tokens,
        "batch_size": 1, "do_sample": False, "seed": SEED,
    }
    run_id = digest(manifest)
    manifest["run_id"] = run_id
    run = args.output / alias / f"{source.lower()}_{split}_{run_id[:16]}"
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(run / ".inferencia.lock", timeout=0):
        path = run / "manifest.json"
        if path.is_file():
            if read_json(path) != manifest:
                raise ValueError(f"El manifiesto existente no coincide: {path}")
        else:
            write_json(path, manifest)
        completed = prediction_records(run, manifest, repair_tail=True)
        print(f"{alias} | {source}/{split} | {len(completed)}/{len(rows)} | contexto {context} | {run}", flush=True)
        with (run / "predictions.jsonl").open("a", encoding="utf-8", newline="\n") as journal:
            for index, row in enumerate(rows, 1):
                if row["example_id"] not in completed:
                    record = generate_row(generator, row, schemas[row["schema_id"]], context,
                                          args.max_new_tokens)
                    record.update(run_id=run_id, model_id=generator.model_id, revision=generator.revision)
                    journal.write(canonical_json(record) + "\n")
                    journal.flush()
                    os.fsync(journal.fileno())
                if index % 25 == 0 or index == len(rows):
                    print(f"{alias} | {source}/{split} | {index}/{len(rows)}", flush=True)


def generate_row(generator, row, schema, context, max_new_tokens):
    messages = messages_for(row["question"], row["dialect"], schema, row.get("evidence"))
    ids = generator.tokenize(messages)
    count = len(ids)
    record = {**row, "input_tokens": count, "output_tokens": 0, "generation_seconds": None,
              "output_raw": None, "sql_generated": None, "error": None}
    if count + max_new_tokens > context:
        return dict(record, status="excluded_context_length",
                    error=f"{count} + {max_new_tokens} > {context} tokens")

    generator.load_weights()
    start = time.monotonic()
    try:
        raw, tokens = generator.generate(ids, max_new_tokens)
    except generator.torch.cuda.OutOfMemoryError:
        record.update(status="generation_oom", error="Memoria CUDA insuficiente.")
    else:
        sql = clean_sql(raw)
        record.update(status="generated" if sql else "empty_output",
                      sql_generated=sql or None, output_raw=raw, output_tokens=tokens)
    if record["status"] == "generation_oom":
        generator.torch.cuda.empty_cache()
    record["generation_seconds"] = time.monotonic() - start
    return record


def run_model(alias, settings, jobs, args, runtime):
    generator = Generator(settings)
    for job in jobs:
        generate_partition(generator, alias, job, args, runtime)


def main():
    models = read_json(SCRIPT_DIR / "modelos.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=Path("master_dataset.parquet"))
    parser.add_argument("--schemas", type=Path, default=Path("schemas.json"))
    parser.add_argument("--output", type=Path, default=Path("resultados_inferencia"))
    parser.add_argument("--model", choices=[*models, "all"], default="qwen-coder")
    parser.add_argument("--dataset", type=str.upper, choices=[*PARTITIONS, "ALL"], default="ALL")
    parser.add_argument("--limit", type=positive_int, help="Número máximo de preguntas por fuente.")
    parser.add_argument("--context-limit", type=positive_int, default=32768)
    parser.add_argument("--max-new-tokens", type=positive_int, default=512)
    parser.add_argument("--use-evidence", action="store_true", help="Incluir la evidencia original de BIRD.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Revisar los datos y la selección en CPU.")
    mode.add_argument("--check-gpu", action="store_true", help="Probar CUDA, BF16 y SDPA sin cargar modelos.")
    args = parser.parse_args()

    if args.check_gpu:
        print(canonical_json(check_gpu()))
        return 0
    jobs = select_data(args)
    aliases = list(models) if args.model == "all" else [args.model]
    for source, split, rows, total, schemas, use_evidence in jobs:
        print(f"{source}/{split}: {len(rows)}/{total} preguntas, {len(schemas)} esquemas, "
              f"evidencia={use_evidence}", flush=True)
    for alias in aliases:
        budget = min(args.context_limit, models[alias]["context"])
        print(f"{alias} | {models[alias]['id']} | contexto hasta {budget} tokens", flush=True)
        if args.max_new_tokens >= budget:
            parser.error(f"--max-new-tokens debe ser menor que {budget} para {alias}.")
    if args.plan:
        return 0

    import torch

    runtime = check_gpu()
    runtime["packages"] = {name: metadata.version(name) for name in (
        "transformers", "accelerate", "huggingface-hub", "tokenizers", "safetensors",
        "pandas", "numpy", "pyarrow", "sqlglot",
    )}
    failed = []
    for alias in aliases:
        try:
            run_model(alias, models[alias], jobs, args, runtime)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"{alias} | Falló la corrida | {exc}", file=sys.stderr, flush=True)
            failed.append(alias)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    if failed:
        print(f"Modelos con errores: {', '.join(failed)}", file=sys.stderr)
    return int(bool(failed))


if __name__ == "__main__":
    sys.exit(main())
