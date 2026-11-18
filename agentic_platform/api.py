from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agents import AgentRuntime
from .core import DATA_DIR, audit, ingest_file, search_documents, sql_query
from .ml import train_models

api = FastAPI(title="Enterprise Agentic Data Intelligence Platform", version="1.0.0")


class IngestRequest(BaseModel):
    path: str
    expected_schema: dict[str, str] | None = Field(default=None, alias="schema")
    batch_size: int = Field(default=5000, ge=1, le=50000)


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=20)


class QueryRequest(BaseModel):
    query: str


class AgentRequest(BaseModel):
    question: str
    approve_python: bool = False


class TrainRequest(BaseModel):
    path: str
    task: str
    target: str = ""
    clusters: int = Field(default=3, ge=2, le=20)


@api.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent_backend": os.getenv("AGENT_BACKEND", "local")}


@api.post("/ingest")
def ingest_endpoint(request: IngestRequest) -> dict[str, Any]:
    try:
        source = Path(request.path).expanduser().resolve()
        if not source.is_relative_to(DATA_DIR.resolve()):
            raise PermissionError("API ingestion is restricted to files inside the data directory")
        result = ingest_file(str(source), request.expected_schema, request.batch_size)
        audit({"action": "ingest", "source": Path(request.path).name, "result": result})
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.post("/search")
def search_endpoint(request: SearchRequest) -> list[dict[str, Any]]:
    try:
        return search_documents(request.query, request.limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.post("/query")
def query_endpoint(request: QueryRequest) -> list[dict[str, Any]]:
    try:
        return sql_query(request.query)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.post("/agent")
def agent_endpoint(request: AgentRequest) -> dict[str, Any]:
    try:
        return AgentRuntime().run(request.question, request.approve_python)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.post("/train")
def train_endpoint(request: TrainRequest) -> dict[str, Any]:
    try:
        source = Path(request.path).expanduser().resolve()
        if not source.is_relative_to(DATA_DIR.resolve()):
            raise PermissionError("API training is restricted to files inside the data directory")
        result = train_models(str(source), request.task, request.target, request.clusters)
        audit({"action": "train", "task": request.task, "result": result})
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


