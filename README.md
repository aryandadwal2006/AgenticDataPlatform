# Enterprise Agentic Data Intelligence & Automation Platform

A minimal, local-first data intelligence platform for structured and unstructured data. It includes chunked ingestion, schema checks, deduplication, DuckDB SQL, document retrieval, TensorFlow/PyTorch baselines, and guarded LangChain/CrewAI agents with an optional IBM watsonx.ai model backend.

The default configuration runs without API keys or hosted services. The default agent is deterministic and local. Hosted model calls are opt-in.

## Features

- **Ingestion:** CSV, TSV, JSONL/NDJSON, Parquet, PDF, HTML, TXT, Markdown, and log files. Tabular inputs are processed in batches. Documents are parsed and split into overlapping chunks.
- **Validation and deduplication:** optional column/type schema checks for tabular files, within-batch duplicate removal, and persistent row/chunk hashes.
- **Storage and query:** DuckDB tables named `data_<filename>` plus document and embedding tables. The SQL tool accepts one read-only `SELECT` against platform-created tables.
- **Retrieval:** cosine-similarity search over local vector representations. Hashing embeddings are the zero-download default. A local Hugging Face Sentence Transformers model can be enabled with environment variables.
- **Machine learning:** held-out classification and regression baselines trained independently in TensorFlow and PyTorch. K-Means clustering reports inertia and silhouette score.
- **Agents:** a local deterministic tool router, a LangChain tool-calling loop backed by `ChatWatsonx`, and a CrewAI agent adapter. Tools cover SQL, Python analysis, and document retrieval.
- **Guardrails:** read-only SQL, AST-restricted Python, explicit Python approval, prompt-injection pattern checks, tool-call limits, CrewAI execution limits, and JSONL audit events.
- **API and container:** FastAPI endpoints and a Docker image.

## Requirements

- Python 3.11
- pip
- Docker is optional

The full requirements file installs TensorFlow CPU, PyTorch, Hugging Face, LangChain, CrewAI, DuckDB, FastAPI, and the file parsers. The ML and agent packages are large; the local ingestion and API paths do not need model credentials.

## Setup

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
New-Item -ItemType Directory -Force data | Out-Null
Copy-Item examples\orders.csv data\orders.csv
Copy-Item examples\operations.md data\operations.md
```

### Linux or macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
mkdir -p data
cp examples/orders.csv data/orders.csv
cp examples/operations.md data/operations.md
```

No `.env` file or API key is needed for the local workflow.

## Run the local workflow

Ingest a structured file and a document:

```bash
python app.py ingest data/orders.csv
python app.py ingest data/operations.md
```

The CSV is stored in DuckDB as `data_orders`. The document is parsed, chunked, and stored in the `documents` table. Re-ingesting identical rows or chunks does not add duplicates.

Run retrieval:

```bash
python app.py search "nightly import duplicate records"
```

Run read-only SQL through the local agent:

```bash
python app.py agent "sql: SELECT COUNT(*) AS row_count FROM data_orders"
```

Run document retrieval through the local agent:

```bash
python app.py agent "search: What should I check after a failed nightly import?"
```

Run supervised baselines. Training reads the input CSV directly and uses an 80/20 held-out split:

```bash
python app.py train examples/orders.csv --task regression --target revenue
python app.py train examples/orders.csv --task classification --target segment
```

Run clustering:

```bash
python app.py train examples/orders.csv --task clustering --clusters 3
```

The classification and regression commands train both a TensorFlow network and a PyTorch network and report test-set metrics. Clustering uses scikit-learn K-Means.

## Start the API

```bash
python app.py serve --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive API docs.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | service status |
| POST | `/ingest` | ingest a local file under `data/` |
| POST | `/search` | retrieve matching document chunks |
| POST | `/query` | execute a read-only SQL query |
| POST | `/agent` | run the configured agent |
| POST | `/train` | train and evaluate ML baselines from a CSV under `data/` |

Example requests:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path":"data/orders.csv","batch_size":1000}'

curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"failed nightly import","limit":5}'

curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"SELECT region, AVG(revenue) AS average_revenue FROM data_orders GROUP BY region"}'

curl -X POST http://127.0.0.1:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"question":"sql: SELECT COUNT(*) AS row_count FROM data_orders"}'
```

For schema validation, provide the required columns and expected types:

```json
{
  "path": "data/orders.csv",
  "batch_size": 1000,
  "schema": {
    "order_id": "integer",
    "region": "string",
    "revenue": "number"
  }
}
```

Supported schema types are `string`, `number`, `integer`, `boolean`, and `datetime`. Schema validation is applied to each batch before it is written. API ingestion and training are restricted to files inside the configured data directory; the CLI can read a local path supplied by the operator.

## Optional Hugging Face embeddings

Hashing embeddings are the default and need no model download. To use a local Sentence Transformers model, set:

```bash
EMBEDDING_BACKEND=hf
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

The model is downloaded from Hugging Face on first use and cached locally. No Hugging Face token is needed for public models. For a fully offline run, keep `EMBEDDING_BACKEND=hashing`.

## Optional IBM watsonx.ai + LangChain

The default `AGENT_BACKEND=local` does not use an LLM. To enable LangChain tool calling with IBM watsonx.ai, provide the credentials in the environment where the app runs:

```bash
AGENT_BACKEND=watsonx
WATSONX_API_KEY=<your-watsonx-api-key>
WATSONX_PROJECT_ID=<your-watsonx-project-id>
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_MODEL_ID=<model-id-enabled-in-your-watsonx-project>
```

`ChatWatsonx.bind_tools()` exposes the SQL, retrieval, and restricted Python functions. The runtime executes tool calls itself so every call passes through the same permission checks and audit path. Use the correct regional URL and an available model ID for your watsonx.ai project. Do not commit credentials.

## Optional CrewAI backend

CrewAI can be selected separately. Configure a model identifier supported by your installed CrewAI/LiteLLM provider adapter and its corresponding environment credentials:

```bash
AGENT_BACKEND=crewai
CREWAI_LLM=<provider/model-id>
WATSONX_API_KEY=<your-watsonx-api-key>
WATSONX_PROJECT_ID=<your-watsonx-project-id>
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

The CrewAI agent receives read-only SQL and retrieval tools plus the same approval-gated Python tool. Set the provider credentials required by your configured CrewAI/LiteLLM model adapter. If provider credentials are not configured, keep `AGENT_BACKEND=local`.

## Permission and safety model

- SQL must be a single `SELECT`, may only reference tables created by the platform, blocks file/extension access functions, applies a 500-row result cap, and uses a DuckDB memory limit.
- Python is disabled unless `approve_python` is explicitly set to `true` on the API request or `--approve-python` is supplied on the CLI. It accepts a small AST allowlist, rejects imports, loops, private attributes, arbitrary function calls, and file/network access, and exposes at most 1,000 rows from the first structured table.
- Prompt-injection checks reject several common instruction-override patterns. Retrieved document content is treated as untrusted data in model instructions.
- LangChain tool loops are capped by `MAX_AGENT_STEPS` (default 5). CrewAI also receives a 5-iteration and 60-second execution limit.
- Tool name, approval decision, and duration are written to `data/audit.jsonl`. Raw prompts and tool arguments are not written to the audit file.

The Python runner is an AST-restricted execution environment, not a hardened operating-system sandbox. The API has no authentication or rate limiting. Keep it on localhost or behind an authenticated gateway, and run untrusted workloads in an isolated container with restricted filesystem and network access. This minimal version exposes read-only data tools; it does not run shell commands or perform write-side infrastructure changes.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `./data` | data, DuckDB, and audit directory |
| `DB_PATH` | `./data/platform.duckdb` | DuckDB database path |
| `AUDIT_PATH` | `./data/audit.jsonl` | JSONL audit path |
| `CHUNK_SIZE` | `900` | document chunk size in characters |
| `CHUNK_OVERLAP` | `120` | overlap between document chunks |
| `EMBEDDING_BACKEND` | `hashing` | `hashing` or `hf` |
| `AGENT_BACKEND` | `local` | `local`, `watsonx`, or `crewai` |
| `MAX_AGENT_STEPS` | `5` | maximum model tool calls |
| `MAX_PYTHON_ROWS` | `1000` | maximum rows exposed to restricted Python |

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover document chunking, prompt-injection pattern detection, restricted Python syntax, and SQL write blocking.

## Docker

Build and run the API locally:

```bash
docker build -t agentic-data-platform .
docker run --rm -p 8000:8000 -v "${PWD}/data:/app/data" agentic-data-platform
```

For PowerShell:

```powershell
docker build -t agentic-data-platform .
docker run --rm -p 8000:8000 -v "${PWD}/data:/app/data" agentic-data-platform
```

The container starts the API on port 8000. The mounted `data/` directory persists the DuckDB database and audit log.
