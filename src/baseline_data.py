"""Prepare one input per question using the project's original partitions."""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src import config
from src.data_loader import build_master_dataset, load_master_dataset
from src.schema_parser import build_schema_catalog, load_schema_catalog, parse_pg_dump_schema

RUN_COLUMNS = ["example_id", "source", "split", "source_split", "db_id",
               "schema_id", "dialect", "question", "evidence", "sql_gold"]


def prepare_data(processed: Path, raw: Path):
    master_path = processed / config.MASTER_DATASET_PATH.name
    schemas_path = processed / config.SCHEMAS_PATH.name
    master = (load_master_dataset(master_path) if master_path.is_file()
              else build_master_dataset(raw_dir=raw, processed_dir=processed))
    catalog = (load_schema_catalog(schemas_path) if schemas_path.is_file()
               else build_schema_catalog(raw_dir=raw, processed_dir=processed,
                                         required_schema_ids=set(master.schema_id)))
    for schema_id, schema in catalog.items():
        if schema.get("pg_schema") and ("foreign_key_groups" not in schema or "view_ddl" not in schema):
            dump = config.raw_file_paths(raw)[2]["dump_sql"]
            if dump.is_file():
                catalog[schema_id] = parse_pg_dump_schema(dump, schema["pg_schema"], schema["db_id"])
    if master.example_id.isna().any() or master.example_id.duplicated().any():
        raise ValueError("El maestro requiere un example_id único por pregunta.")
    return master, catalog


def select_questions(master, source: str, split: str, limit: int | None, seed: int):
    subset = master.loc[(master.source == source) & (master.split == split), RUN_COLUMNS]
    subset = subset.sort_values("example_id")
    total = len(subset)
    if not total or (limit is not None and limit < 1):
        raise ValueError(f"Partición vacía o límite inválido para {source}/{split}: {limit}")
    if limit is not None and limit < total:
        subset = subset.sample(n=limit, random_state=seed)
    rows = subset.sort_values("example_id").astype(object)
    return rows.where(pd.notna(rows), None).to_dict("records"), total


def full_schema(schema: dict) -> str:
    if not schema["tables"] or not schema["ddl"]:
        raise ValueError(f"Esquema vacío: {schema['schema_id']}")
    if schema["dialect"] == "sqlite":
        return schema["ddl"]
    if schema.get("views") and "view_ddl" not in schema:
        raise ValueError("El catálogo anterior solo contiene los nombres de vistas de Catastro. "
                         "Conserva el dump SQL en RAW para recuperar sus definiciones.")

    def identifier(name):
        return name if re.fullmatch("[a-z_][a-z0-9_]*", name) else '"' + name.replace('"', '""') + '"'

    columns = defaultdict(list)
    keys = defaultdict(list)
    for column in schema["columns"]:
        columns[column["table_name"]].append(f"{identifier(column['name_original'])} {column['column_type']}")
    for pk in schema["primary_keys"]:
        keys[pk["table"]].append("PK(" + ",".join(map(identifier, pk["columns"])) + ")")
    foreign_keys = schema.get("foreign_key_groups")
    if foreign_keys is None:
        foreign_keys = [{"from_table": fk["from_table"], "from_columns": identifier(fk["from_column"]),
                         "to_table": fk["to_table"], "to_columns": identifier(fk["to_column"])}
                        for fk in schema["foreign_keys"]]
    for fk in foreign_keys:
        keys[fk["from_table"]].append(
            f"FK({fk['from_columns']})->{identifier(fk['to_table'])}({fk['to_columns']})"
        )
    lines = [f"PostgreSQL schema: {schema['pg_schema']}. "
             "Each line is table(column type,...); PK is the primary key, FK references another table."]
    for table in schema["tables"]:
        name = table["name_original"]
        lines.append(f"{identifier(name)}({','.join(columns[name])})" + "".join(f";{key}" for key in keys[name]))
    return "\n".join([*lines, *schema.get("view_ddl", [])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--raw", type=Path, default=config.RAW_DIR)
    args = parser.parse_args()
    master, catalog = prepare_data(args.processed, args.raw)
    print(master.groupby(["source", "split"]).agg(
        questions=("example_id", "size"), databases=("db_id", "nunique")
    ).to_string())
    print(f"Esquemas: {len(catalog)}. Referencias ausentes: {sorted(set(master.schema_id) - catalog.keys())}")


if __name__ == "__main__":
    main()
