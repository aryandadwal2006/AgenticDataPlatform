from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "platform.duckdb"))
AUDIT_PATH = Path(os.getenv("AUDIT_PATH", DATA_DIR / "audit.jsonl"))
MAX_PYTHON_ROWS = int(os.getenv("MAX_PYTHON_ROWS", "1000"))
MAX_PYTHON_CHARS = int(os.getenv("MAX_PYTHON_CHARS", "2000"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

INJECTION_PATTERNS = (
    r"ignore (all |the )?(previous|prior) instructions",
    r"reveal (the )?(system|developer) prompt",
    r"exfiltrat(e|ion)",
    r"execute (a )?(shell|terminal|command)",
    r"bypass (the )?(safety|permission|policy)"
)

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> duckdb.DuckDBPyConnection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute(
        "CREATE TABLE IF NOT EXISTS documents ("
        "id BIGINT PRIMARY KEY, source VARCHAR, content VARCHAR, metadata JSON, content_hash VARCHAR UNIQUE)"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS embeddings ("
        "document_id BIGINT, backend VARCHAR, vector DOUBLE[], PRIMARY KEY(document_id, backend))"
    )
    return con


def audit(event: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"time": utc_now(), "event_id": str(uuid.uuid4()), **event}
    with AUDIT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def safe_table_name(path: Path) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_]+", "_", path.stem).strip("_").lower() or "dataset"
    return f"data_{stem[:48]}"


def validate_frame(frame: pd.DataFrame, schema: dict[str, str] | None) -> None:
    if frame.empty:
        return
    if frame.columns.duplicated().any():
        raise ValueError("duplicate column names are not allowed")
    if not schema:
        return
    missing = set(schema) - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    supported = {"string", "number", "integer", "boolean", "datetime"}
    for column, expected in schema.items():
        kind = expected.lower()
        if kind not in supported:
            raise ValueError(f"unsupported schema type for {column}: {expected}")
        series = frame[column]
        valid = {
            "string": pd.api.types.is_string_dtype(series) or series.dtype == object,
            "number": pd.api.types.is_numeric_dtype(series),
            "integer": pd.api.types.is_integer_dtype(series),
            "boolean": pd.api.types.is_bool_dtype(series),
            "datetime": pd.api.types.is_datetime64_any_dtype(series),
        }[kind]
        if not valid:
            raise ValueError(f"column {column} must have type {kind}")


def coerce_schema(frame: pd.DataFrame, schema: dict[str, str] | None) -> pd.DataFrame:
    if not schema:
        return frame
    frame = frame.copy()
    for column, expected in schema.items():
        if column not in frame.columns:
            continue
        kind = expected.lower()
        if kind == "string":
            frame[column] = frame[column].astype("string")
        elif kind in {"number", "integer"}:
            values = pd.to_numeric(frame[column], errors="raise")
            if kind == "integer" and not np.equal(np.mod(values.dropna().astype(float), 1), 0).all():
                raise ValueError(f"column {column} contains non-integer values")
            frame[column] = values.astype("Int64" if kind == "integer" else "float64")
        elif kind == "boolean":
            values = frame[column].map(
                lambda value: value if pd.isna(value) or isinstance(value, (bool, np.bool_))
                else {"true": True, "false": False, "1": True, "0": False}.get(str(value).lower())
            )
            if values.isna().sum() > frame[column].isna().sum():
                raise ValueError(f"column {column} contains invalid boolean values")
            frame[column] = values.astype("boolean")
        elif kind == "datetime":
            frame[column] = pd.to_datetime(frame[column], errors="raise", utc=True)
    return frame


def _insert_frame(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame, table: str, source: str) -> int:
    if frame.empty:
        return 0
    frame = frame.copy().drop_duplicates()
    row_hashes = pd.util.hash_pandas_object(frame.astype(str), index=False).astype(str)
    frame["__source"] = source
    frame["__ingested_at"] = utc_now()
    frame["__row_hash"] = row_hashes
    frame = frame.drop_duplicates(subset=["__row_hash"])
    con.register("_incoming", frame)
    exists = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()[0]
    if not exists:
        con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM _incoming')
    else:
        existing = {row[1] for row in con.execute(f'PRAGMA table_info("{table}")').fetchall()}
        incoming = set(frame.columns)
        if existing != incoming:
            con.unregister("_incoming")
            raise ValueError(f"schema mismatch for {table}; incoming columns must match existing columns")
        con.execute(
            f'INSERT INTO "{table}" SELECT i.* FROM _incoming i '
            f'WHERE NOT EXISTS (SELECT 1 FROM "{table}" t WHERE t.__row_hash = i.__row_hash)'
        )
    inserted = con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
    con.unregister("_incoming")
    return int(inserted)


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + size // 2:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return [chunk for chunk in chunks if chunk]


def extract_documents(path: Path) -> list[tuple[str, dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return [(page.extract_text() or "", {"page": index + 1}) for index, page in enumerate(reader.pages)]
    if suffix in {".html", ".htm"}:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return [(soup.get_text(" ", strip=True), {"kind": "html"})]
    if suffix in {".txt", ".log", ".md"}:
        return [(path.read_text(encoding="utf-8", errors="ignore"), {"kind": suffix[1:]})]
    raise ValueError(f"unsupported file type: {suffix}")


def ingest_file(path_value: str, schema: dict[str, str] | None = None, batch_size: int = 5000) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    con = connect()
    try:
        if suffix in {".csv", ".tsv", ".jsonl", ".ndjson", ".parquet"}:
            table = safe_table_name(path)
            if suffix == ".parquet":
                import pyarrow.parquet as parquet
                batches = (batch.to_pandas() for batch in parquet.ParquetFile(path).iter_batches(batch_size=batch_size))
            elif suffix in {".jsonl", ".ndjson"}:
                batches = pd.read_json(path, lines=True, chunksize=batch_size)
            else:
                batches = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", chunksize=batch_size)
            total = 0
            for batch in batches:
                batch = coerce_schema(batch, schema)
                validate_frame(batch, schema)
                before = con.execute(
                    "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
                ).fetchone()[0]
                prior = int(con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]) if before else 0
                _insert_frame(con, batch, table, path.name)
                now = int(con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
                total += now - prior
            return {"kind": "table", "table": table, "rows_inserted": total}

        documents = extract_documents(path)
        inserted = 0
        next_id = int(con.execute("SELECT coalesce(max(id), 0) + 1 FROM documents").fetchone()[0])
        for raw_text, metadata in documents:
            for chunk in split_text(raw_text):
                digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                result = con.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING RETURNING id",
                    [next_id, path.name, chunk, json.dumps(metadata), digest],
                ).fetchone()
                if result:
                    inserted += 1
                    next_id += 1
        return {"kind": "documents", "chunks_inserted": inserted}
    finally:
        con.close()


class Embedder:
    def __init__(self) -> None:
        self.kind = os.getenv("EMBEDDING_BACKEND", "hashing").lower()
        if self.kind == "hf":
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            self.backend = f"hf:{model_name}"
            self.model = SentenceTransformer(model_name)
        else:
            from sklearn.feature_extraction.text import HashingVectorizer
            self.kind = "hashing"
            self.backend = "hashing:384"
            self.model = HashingVectorizer(n_features=384, alternate_sign=False, norm="l2")

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.kind == "hf":
            return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)
        return self.model.transform(texts).toarray().astype(np.float32)


def search_documents(query: str, limit: int = 5) -> list[dict[str, Any]]:
    con = connect()
    try:
        rows = con.execute("SELECT id, source, content, metadata FROM documents ORDER BY id").fetchall()
        if not rows:
            return []
        embedder = Embedder()
        backend = embedder.backend
        cached = {
            row[0]: np.asarray(row[1], dtype=np.float32)
            for row in con.execute("SELECT document_id, vector FROM embeddings WHERE backend = ?", [backend]).fetchall()
        }
        missing = [row for row in rows if row[0] not in cached]
        if missing:
            vectors = embedder.encode([row[2] for row in missing])
            for row, vector in zip(missing, vectors):
                con.execute(
                    "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)",
                    [row[0], backend, vector.astype(float).tolist()],
                )
                cached[row[0]] = vector
        q = embedder.encode([query])[0]
        ranked = []
        for row in rows:
            vector = cached[row[0]]
            denom = float(np.linalg.norm(q) * np.linalg.norm(vector))
            score = float(np.dot(q, vector) / denom) if denom else 0.0
            ranked.append({"id": row[0], "source": row[1], "content": row[2], "metadata": json.loads(row[3]), "score": score})
        return sorted(ranked, key=lambda item: item["score"], reverse=True)[: max(1, min(limit, 20))]
    finally:
        con.close()


def sql_query(query: str) -> list[dict[str, Any]]:
    statement = query.strip()
    if not re.match(r"(?is)^select\b", statement) or ";" in statement.rstrip(";"):
        raise PermissionError("only one read-only SELECT statement is allowed")
    if "--" in statement or "/*" in statement or "*/" in statement:
        raise PermissionError("SQL comments are not allowed")
    if re.search(r"(?i)\b(?:from|join)\s+['\"]", statement):
        raise PermissionError("file-backed SQL sources are not allowed")
    if re.search(r"(?i)\b(insert|update|delete|drop|alter|attach|detach|copy|export|import|install|load|call|pragma|set|reset|read_[a-z0-9_]*|[a-z0-9_]*_scan|parquet_scan|glob|httpfs|range|generate_series|unnest|query_table)\b", statement):
        raise PermissionError("query contains a blocked SQL operation")
    referenced = re.findall(r"(?i)\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", statement)
    con = connect()
    try:
        allowed_tables = {
            row[0] for row in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        }
        if any(table.lower() not in allowed_tables for table in referenced):
            raise PermissionError("queries may only read tables created by this platform")
        con.execute("SET memory_limit='256MB'")
        con.execute("SET threads=2")
        bounded = statement.rstrip().rstrip(";")
        limit_match = re.search(r"(?i)\blimit\s+(\d+)\s*$", bounded)
        if limit_match:
            bounded = bounded[:limit_match.start()] + f"LIMIT {min(int(limit_match.group(1)), 500)}"
        else:
            bounded += " LIMIT 500"
        return con.execute(bounded).df().head(500).to_dict(orient="records")
    finally:
        con.close()


ALLOWED_AST = (
    ast.Module, ast.Expression, ast.Expr, ast.Assign, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.List, ast.Tuple, ast.Dict, ast.Subscript, ast.Slice, ast.BinOp, ast.UnaryOp,
    ast.BoolOp, ast.Compare, ast.Call, ast.Attribute, ast.Add, ast.Sub, ast.Mult,
    ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
)
SAFE_FUNCTIONS = {"len": len, "sum": sum, "min": min, "max": max, "round": round, "abs": abs}
SAFE_METHODS = {"head", "tail", "describe", "value_counts", "mean", "sum", "min", "max", "nunique"}
SAFE_NUMPY = {"mean", "median", "std", "var", "min", "max", "unique", "percentile", "sqrt"}


class PythonGuard(ast.NodeVisitor):
    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, ALLOWED_AST):
            raise PermissionError(f"python construct is not allowed: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("_") or node.id not in {"df", "np", "pd", *SAFE_FUNCTIONS, "result"}:
            raise PermissionError(f"name is not allowed: {node.id}")

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, (int, float)) and abs(node.value) > 1_000_000:
            raise PermissionError("numeric literals are too large")
        if isinstance(node.value, str) and len(node.value) > 10000:
            raise PermissionError("string literals are too large")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Pow) and not (
            isinstance(node.right, ast.Constant)
            and isinstance(node.right.value, (int, float))
            and 0 <= node.right.value <= 8
        ):
            raise PermissionError("exponent is not allowed")
        if isinstance(node.op, ast.Mult) and isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)) and abs(node.right.value) > 1000:
            raise PermissionError("multiplication factor is too large")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            raise PermissionError("private attributes are not allowed")
        if isinstance(node.value, ast.Name) and node.value.id == "df" and node.attr not in SAFE_METHODS | {"shape", "columns", "dtypes"}:
            raise PermissionError(f"dataframe attribute is not allowed: {node.attr}")
        if isinstance(node.value, ast.Name) and node.value.id == "np" and node.attr not in SAFE_NUMPY:
            raise PermissionError(f"numpy attribute is not allowed: {node.attr}")
        if isinstance(node.value, ast.Name) and node.value.id not in {"df", "np"}:
            raise PermissionError("attribute access is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id not in SAFE_FUNCTIONS:
                raise PermissionError(f"function is not allowed: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            allowed = node.func.attr in SAFE_METHODS or (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "np"
                and node.func.attr in SAFE_NUMPY
            )
            if not allowed:
                raise PermissionError(f"method is not allowed: {node.func.attr}")
        else:
            raise PermissionError("dynamic function calls are not allowed")
        self.generic_visit(node)


def safe_python(code: str) -> str:
    if len(code) > MAX_PYTHON_CHARS:
        raise ValueError("python input is too long")
    tree = ast.parse(code, mode="exec")
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ValueError("python expression is too complex")
    PythonGuard().visit(tree)
    con = connect()
    try:
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main' AND table_name LIKE 'data_%' ORDER BY table_name"
        ).fetchall()
        if not tables:
            raise ValueError("no structured data has been ingested")
        table = tables[0][0]
        df = con.execute(f'SELECT * FROM "{table}" LIMIT {MAX_PYTHON_ROWS}').df()
    finally:
        con.close()
    scope = {"df": df, "np": np, "pd": pd, **SAFE_FUNCTIONS}
    exec(compile(tree, "<safe-python>", "exec"), {"__builtins__": {}}, scope)
    output = scope.get("result")
    if output is None:
        output = {key: value for key, value in scope.items() if key not in {"df", "np", "pd", *SAFE_FUNCTIONS}}
    if isinstance(output, (pd.DataFrame, pd.Series)):
        return output.head(100).to_string()
    return str(output)[:8000]


def is_prompt_injection(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in INJECTION_PATTERNS)


