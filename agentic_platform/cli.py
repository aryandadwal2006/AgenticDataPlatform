from __future__ import annotations

import argparse
import json

from .agents import AgentRuntime
from .api import api
from .core import ingest_file, search_documents
from .ml import train_models


def main() -> None:
    parser = argparse.ArgumentParser(description="local-first enterprise data intelligence platform")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    ingest = sub.add_parser("ingest")
    ingest.add_argument("path")

    search = sub.add_parser("search")
    search.add_argument("query")

    train = sub.add_parser("train")
    train.add_argument("path")
    train.add_argument("--task", choices=["classification", "regression", "clustering"], required=True)
    train.add_argument("--target", default="")
    train.add_argument("--clusters", type=int, default=3)

    agent = sub.add_parser("agent")
    agent.add_argument("question")
    agent.add_argument("--approve-python", action="store_true")

    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        uvicorn.run(api, host=args.host, port=args.port)
    elif args.command == "ingest":
        print(json.dumps(ingest_file(args.path), indent=2))
    elif args.command == "search":
        print(json.dumps(search_documents(args.query), indent=2))
    elif args.command == "train":
        if args.task != "clustering" and not args.target:
            parser.error("--target is required for classification and regression")
        print(json.dumps(train_models(args.path, args.task, args.target, args.clusters), indent=2))
    elif args.command == "agent":
        print(json.dumps(AgentRuntime().run(args.question, args.approve_python), indent=2))


if __name__ == "__main__":
    main()
