"""Prompt completo, inferencia con Qwen y registro incremental por pregunta."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import time
from pathlib import Path
from uuid import uuid4

SYSTEM_PROMPT = (
    "Generate exactly one read-only SQL query answering the question. "
    "Use only the supplied schema and the specified SQL dialect. "
    "Return SQL only, without Markdown or explanation."
)
FENCE = re.compile(r"\s*```(?:sql|sqlite|postgresql|postgres)?[ \t]*\r?\n(.*?)\r?\n?[ \t]*```\s*", re.I | re.S)
PACKAGE_NAMES = ("torch", "transformers", "accelerate", "bitsandbytes", "huggingface-hub",
                 "tokenizers", "safetensors", "numpy", "pandas", "pyarrow", "sqlglot", "openpyxl")
MANIFEST_KEYS = ("example_id", "db_id", "schema_id")


def messages_for(question: str, dialect: str, schema: str, evidence: str | None = None):
    dialect_name = {"sqlite": "SQLite", "postgres": "PostgreSQL with PostGIS"}[dialect]
    user = f"Dialect: {dialect_name}\n\nComplete schema:\n{schema}\n\nQuestion:\n{question}"
    if evidence:
        user += f"\n\nDataset evidence:\n{evidence}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def clean_sql(raw: str) -> str:
    fenced = FENCE.fullmatch(raw)
    return (fenced[1] if fenced else raw).strip()


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temp.write_text(content, encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path: Path, payload):
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def prediction_records(run: Path, manifest: dict, *, repair_tail: bool = False) -> dict:
    path = run / "predictions.jsonl"
    if not path.exists():
        return {}
    expected = {item["example_id"] for item in manifest["selected"]}
    records = {}
    with path.open("r+b" if repair_tail else "rb") as fh:
        while line := fh.readline():
            complete = line.endswith(b"\n")
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError):
                if complete:
                    raise
                if repair_tail:
                    fh.truncate(fh.tell() - len(line))
                break
            example_id = row["example_id"]
            if row["run_id"] != manifest["run_id"] or example_id not in expected or example_id in records:
                raise ValueError(f"Predicción incompatible o duplicada en {path}: {example_id}")
            records[example_id] = row
            if not complete and repair_tail:
                fh.write(b"\n")
    return records


def parse_eval_settings(settings):
    timeout = float(settings["timeout_seconds"])
    max_rows = int(settings["max_rows"])
    if not 0 < timeout <= 3600 or not 0 < max_rows <= 1_000_000:
        raise ValueError("timeout_seconds debe estar en (0, 3600] y max_rows en [1, 1000000].")
    return {"timeout_seconds": timeout, "max_rows": max_rows}


class QwenGenerator:
    def __init__(self, settings: dict):
        import torch
        from packaging.version import Version
        from transformers import AutoConfig, AutoTokenizer

        if Version(torch.__version__) < Version("2.4") or not torch.cuda.is_available() or not torch.version.cuda:
            raise RuntimeError("Activa una GPU NVIDIA en Colab con PyTorch CUDA >= 2.4.")
        capability = torch.cuda.get_device_capability()
        if capability[0] < 6:
            raise RuntimeError("La cuantización NF4 requiere una GPU NVIDIA con capacidad CUDA >= 6.0.")
        if settings["batch_size"] != 1 or settings["do_sample"] is not False:
            raise ValueError("El baseline usa batch_size=1 y do_sample=false.")
        self.torch = torch
        self.dtype = (torch.bfloat16 if capability[0] >= 8 and torch.cuda.is_bf16_supported()
                      else torch.float16)
        self.model_id = settings["model_id"]
        self.revision = settings["revision"]
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("revision debe ser el SHA completo del checkpoint en Hugging Face.")
        self.max_new = int(settings["max_new_tokens"])
        model_config = AutoConfig.from_pretrained(self.model_id, revision=self.revision)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, verbose=False
        )
        self.model = None
        self.runtime = {
            "python": platform.python_version(),
            "packages": {name: metadata.version(name) for name in PACKAGE_NAMES},
            "gpu": torch.cuda.get_device_name(),
            "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda": torch.version.cuda,
            "compute_dtype": str(self.dtype),
            "model_context": model_config.max_position_embeddings,
        }

    def tokenize(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", truncation=False,
        )

    def load(self):
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        if self.model is None:
            quantization = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=self.dtype,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, revision=self.revision, use_safetensors=True,
                quantization_config=quantization, dtype=self.dtype,
                device_map={"": 0}, attn_implementation="sdpa",
            ).eval()

    def generate(self, inputs):
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, num_beams=1, max_new_tokens=self.max_new,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_ids = output[0, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True), new_ids.numel()


def infer_partition(master, catalog, job: dict, settings: dict, generator: QwenGenerator,
                    output_root: Path) -> Path:
    from src.baseline_data import full_schema, select_questions

    source, split = job["source"], job["split"]
    if source == "CATASTRO" and split != "test_transfer":
        raise ValueError("Catastro solo se ejecuta en test_transfer.")
    budget = int(job.get("context_budget", settings["context_budget"]))
    model_limit = generator.runtime["model_context"]
    if not 0 < generator.max_new < budget <= model_limit:
        raise ValueError(f"Se requiere 0 < max_new_tokens < context_budget <= {model_limit}.")
    rows, total = select_questions(master, source, split, job["limit"], settings["seed"])
    schemas = {schema_id: full_schema(catalog[schema_id])
               for schema_id in sorted({row["schema_id"] for row in rows})}
    use_evidence = bool(settings["use_evidence"] and source == "BIRD")
    manifest = {
        "source": source, "split": split, "partition_size": total,
        "selected": [{key: row[key] for key in MANIFEST_KEYS} for row in rows],
        "rows_sha256": digest(rows), "schemas_sha256": digest(schemas),
        "prompt_and_inference_sha256": source_hash(Path(__file__)),
        "evaluation_sha256": source_hash(Path(__file__).with_name("baseline_evaluation.py")),
        "model_id": generator.model_id, "revision": generator.revision,
        "runtime": generator.runtime, "quantization": "NF4 with double quantization",
        "do_sample": False, "batch_size": 1, "seed": settings["seed"],
        "context_budget": budget, "max_new_tokens": generator.max_new,
        "use_evidence": use_evidence, "evaluation": parse_eval_settings(settings["evaluation"]),
    }
    manifest["run_id"] = digest(manifest)
    run = output_root / f"{source.lower()}_{split}_{manifest['run_id'][:16]}"
    if source == "CATASTRO":
        freeze_path = output_root / "catastro_transfer_lock.json"
        if freeze_path.is_file() and read_json(freeze_path) != manifest:
            raise ValueError(f"Catastro ya tiene una configuración congelada en {freeze_path}.")
        if not freeze_path.is_file():
            write_json(freeze_path, manifest)
    manifest_path = run / "manifest.json"
    if manifest_path.is_file() and read_json(manifest_path) != manifest:
        raise ValueError(f"Manifiesto incompatible: {manifest_path}")
    if not manifest_path.is_file():
        write_json(manifest_path, manifest)
    completed = prediction_records(run, manifest, repair_tail=True)
    print(f"{source}/{split} | {len(completed)}/{len(rows)} registrados | {run}", flush=True)
    with (run / "predictions.jsonl").open("a", encoding="utf-8", newline="\n") as journal:
        for index, row in enumerate(rows, 1):
            if row["example_id"] in completed:
                continue
            messages = messages_for(row["question"], row["dialect"], schemas[row["schema_id"]],
                                    row["evidence"] if use_evidence else None)
            inputs = generator.tokenize(messages)
            count = inputs["input_ids"].shape[-1]
            record = {**row, "run_id": manifest["run_id"], "use_evidence": use_evidence,
                      "input_tokens": count, "output_tokens": 0, "generation_seconds": None,
                      "status": "excluded_context_length", "error": None,
                      "sql_generated": None, "output_raw": None}
            if count + generator.max_new <= budget:
                generator.load()
                start = time.monotonic()
                try:
                    raw, output_count = generator.generate(inputs)
                    sql = clean_sql(raw)
                    record.update(output_raw=raw, output_tokens=output_count, sql_generated=sql,
                                  status="generated" if sql else "empty_output")
                except generator.torch.cuda.OutOfMemoryError:
                    record.update(status="generation_oom", error="Memoria CUDA insuficiente.")
                except RuntimeError as exc:
                    record.update(status="generation_error", error=str(exc))
                if record["status"] == "generation_oom":
                    generator.torch.cuda.empty_cache()
                record["generation_seconds"] = time.monotonic() - start
            else:
                record["error"] = f"{count} + {generator.max_new} > {budget} tokens"
            journal.write(canonical_json(record) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            if index % 25 == 0 or index == len(rows):
                print(f"{source}/{split} | {index}/{len(rows)}", flush=True)
    return run
