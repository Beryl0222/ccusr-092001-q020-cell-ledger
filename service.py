"""跨物种细胞图谱实验账本服务。

提供健康检查与实验账本的 REST 接口，状态持久化在只追加的 JSONL 事件日志中
（默认 ``ledger.jsonl``，可用 ``--data`` 或环境变量 ``LEDGER_DATA`` 指定）。
"""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from ledger_core import (
    ConflictError,
    Ledger,
    LedgerError,
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
)

SERVICE_ID = "cross-species-cell-ledger"

_ERROR_STATUS = {
    ValidationError: 400,
    NotFoundError: 404,
    PermissionDeniedError: 403,
    ConflictError: 409,
}


def health():
    """返回稳定的服务身份。"""
    return {"status": "ok", "service": SERVICE_ID}


def build_handler(ledger: Ledger) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        """账本 HTTP 路由。"""

        server_version = "CrossSpeciesCellLedger/1.0"

        # -- 基础 ----------------------------------------------------------

        def _send(self, status: int, payload) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length == 0:
                return {}
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValidationError(f"请求体不是合法 JSON: {exc}") from exc
            if not isinstance(data, dict):
                raise ValidationError("请求体必须是 JSON 对象")
            return data

        def _handle_error(self, exc: LedgerError) -> None:
            status = _ERROR_STATUS.get(type(exc), 500)
            self._send(status, {"error": type(exc).__name__, "message": str(exc)})

        def log_message(self, *_args):
            return

        # -- 路由 ----------------------------------------------------------

        def do_GET(self):
            path = urlparse(self.path).path.rstrip("/") or "/"
            try:
                if path == "/health":
                    self._send(200, health())
                elif path == "/datasets":
                    self._send(200, {"datasets": ledger.list_datasets()})
                elif path == "/runs":
                    self._send(200, {"runs": ledger.list_runs()})
                elif path == "/versions":
                    self._send(200, {"versions": ledger.list_versions()})
                elif path.startswith("/datasets/"):
                    self._route_dataset_get(path)
                elif path.startswith("/runs/"):
                    parts = path.split("/")
                    self._send(200, ledger.get_run(parts[2]))
                elif path.startswith("/versions/"):
                    parts = path.split("/")
                    if len(parts) == 3:
                        self._send(200, ledger.get_version(parts[2]))
                    else:
                        self.send_error(404)
                elif path.startswith("/conclusions/"):
                    parts = path.split("/")
                    if len(parts) == 3:
                        self._send(200, ledger.get_conclusion(parts[2]))
                    elif len(parts) == 4 and parts[3] == "trace":
                        self._send(200, ledger.trace_conclusion(parts[2]))
                    else:
                        self.send_error(404)
                else:
                    self.send_error(404)
            except LedgerError as exc:
                self._handle_error(exc)

        def _route_dataset_get(self, path: str) -> None:
            parts = path.split("/")
            # /datasets/<id> 或 /datasets/<id>/affected
            if len(parts) == 3:
                self._send(200, ledger.get_dataset(parts[2]))
            elif len(parts) == 4 and parts[3] == "affected":
                self._send(200, ledger.affected_report(parts[2]))
            else:
                self.send_error(404)

        def do_POST(self):
            path = urlparse(self.path).path.rstrip("/") or "/"
            try:
                data = self._read_json()
                route = self._post_routes().get(path)
                if route is None:
                    self.send_error(404)
                    return
                self._send(201, route(data))
            except KeyError as exc:
                self._handle_error(ValidationError(f"缺少必填字段: {exc}"))
            except LedgerError as exc:
                self._handle_error(exc)

        def _post_routes(self) -> dict:
            L = ledger
            return {
                "/datasets": lambda d: L.register_dataset(
                    project_id=d["project_id"],
                    species=d["species"],
                    license=d["license"],
                    tissue=d.get("tissue"),
                    disease_state=d.get("disease_state"),
                    license_expires_at=d.get("license_expires_at"),
                    dataset_id=d.get("dataset_id"),
                ),
                "/mappings": lambda d: L.register_mapping(
                    name=d["name"],
                    source_species=d["source_species"],
                    target_species=d["target_species"],
                    version=d["version"],
                    covered_species=d["covered_species"],
                    mapping_id=d.get("mapping_id"),
                ),
                "/models": lambda d: L.register_model(
                    name=d["name"],
                    code_version=d["code_version"],
                    params=d.get("params"),
                    model_id=d.get("model_id"),
                ),
                "/runs": lambda d: L.submit_run(
                    project_id=d["project_id"],
                    run_type=d["run_type"],
                    dataset_ids=d["dataset_ids"],
                    model_id=d["model_id"],
                    code_version=d["code_version"],
                    params=d.get("params"),
                    mapping_id=d.get("mapping_id"),
                    derived_from=d.get("derived_from"),
                    run_id=d.get("run_id"),
                ),
                "/queries": lambda d: L.create_query(
                    project_id=d["project_id"],
                    version_id=d["version_id"],
                    inputs=d["inputs"],
                    model_id=d.get("model_id"),
                    mapping_id=d.get("mapping_id"),
                    query_id=d.get("query_id"),
                ),
                "/conclusions": lambda d: L.create_conclusion(
                    project_id=d["project_id"],
                    query_id=d["query_id"],
                    claim=d["claim"],
                    inputs=d["inputs"],
                    evidence_level=d.get("evidence_level", "模型推断"),
                    conclusion_id=d.get("conclusion_id"),
                ),
                "/annotations": lambda d: L.add_annotation(
                    target_id=d["target_id"], author=d["author"], text=d["text"]
                ),
                "/validations": lambda d: L.record_validation(
                    conclusion_id=d["conclusion_id"],
                    evidence_level=d["evidence_level"],
                    method=d["method"],
                    result=d["result"],
                ),
                "/grants": lambda d: L.issue_grant(
                    dataset_id=d["dataset_id"],
                    project_id=d["project_id"],
                    scope=d["scope"],
                    expires_at=d.get("expires_at"),
                ),
                "/grants/revoke": lambda d: L.revoke_grant(
                    dataset_id=d["dataset_id"],
                    project_id=d["project_id"],
                    scope=d["scope"],
                ),
            }

        def do_PUT(self):
            self._route_lifecycle()

        def do_PATCH(self):
            self._route_lifecycle()

        def _route_lifecycle(self) -> None:
            path = urlparse(self.path).path.rstrip("/") or "/"
            try:
                data = self._read_json()
                parts = path.split("/")
                # /runs/<id>/(start|fail|retry|complete)
                if (
                    len(parts) == 4
                    and parts[1] == "runs"
                    and parts[3] in {"start", "fail", "retry", "complete"}
                ):
                    run_id = parts[2]
                    action = parts[3]
                    result = {
                        "start": lambda: ledger.start_run(run_id),
                        "fail": lambda: ledger.fail_run(run_id, data.get("reason")),
                        "retry": lambda: ledger.retry_run(run_id),
                        "complete": lambda: ledger.complete_run(run_id),
                    }[action]()
                    self._send(200, result)
                # /datasets/<id>/qc, /datasets/<id>/revoke
                elif len(parts) == 4 and parts[1] == "datasets":
                    if parts[3] == "qc":
                        self._send(
                            200,
                            ledger.record_qc(
                                parts[2], data["metrics"], data["passed"]
                            ),
                        )
                    elif parts[3] == "revoke":
                        self._send(200, ledger.revoke_dataset_license(
                            parts[2], data.get("reason")
                        ))
                    else:
                        self.send_error(404)
                # /versions/<id>/derive
                elif len(parts) == 4 and parts[1] == "versions" and parts[3] == "derive":
                    self._send(201, ledger.derive_version(
                        base_version_id=parts[2],
                        project_id=data["project_id"],
                        run_type=data["run_type"],
                        dataset_ids=data["dataset_ids"],
                        code_version=data["code_version"],
                        params=data.get("params"),
                        mapping_id=data.get("mapping_id"),
                    ))
                else:
                    self.send_error(404)
            except KeyError as exc:
                self._handle_error(ValidationError(f"缺少必填字段: {exc}"))
            except LedgerError as exc:
                self._handle_error(exc)

    return Handler


def create_server(port: int, data_path: str | None) -> ThreadingHTTPServer:
    ledger = Ledger(data_path)
    return ThreadingHTTPServer(("0.0.0.0", port), build_handler(ledger))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="跨物种细胞图谱实验账本")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data", default=os.environ.get("LEDGER_DATA", "ledger.jsonl"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        Ledger(None)  # 加载并校验 domain.json 词表
        print("基础检查通过")
    else:
        create_server(args.port, args.data).serve_forever()
