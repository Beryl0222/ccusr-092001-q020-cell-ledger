"""跨物种细胞图谱实验账本服务。

只追加事件账本的 HTTP 接口。所有写操作经领域闸门校验后追加事件，
并实时写入 JSONL 审计文件（--store）；服务重启时重放恢复并校验哈希链。

运行::

    python3 service.py --check            # 配置与审计链自检
    python3 service.py --port 8000        # 启动服务（默认 ledger.jsonl）
    python3 -m unittest -v                # 运行测试

错误响应形如 ``{"error": {"code": "...", "message": "...", "details": {}}}``。
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ledger import (
    Ledger,
    LedgerError,
    load_vocab,
    replay,
)
from store import JsonlStore

SERVICE_ID = "cross-species-cell-ledger"


def health():
    """返回稳定的服务身份。"""
    return {"status": "ok", "service": SERVICE_ID}


# 路由：(方法, 正则) -> (ledger 方法名, 路径参数名列表)
ROUTES_POST = [
    (r"^/projects$", "register_project", []),
    (r"^/datasets$", "register_dataset", []),
    (r"^/datasets/(?P<dataset_id>[^/]+)/qc$", "record_qc", ["dataset_id"]),
    (r"^/grants$", "grant_access", []),
    (r"^/grants/(?P<grant_id>[^/]+)/revoke$", "revoke_access", ["grant_id"]),
    (r"^/mappings$", "register_mapping", []),
    (r"^/code-versions$", "register_code_version", []),
    (r"^/runs$", "start_run", []),
    (r"^/runs/(?P<run_id>[^/]+)/fail$", "fail_run", ["run_id"]),
    (r"^/runs/(?P<run_id>[^/]+)/retry$", "retry_run", ["run_id"]),
    (r"^/runs/(?P<run_id>[^/]+)/complete$", "complete_run", ["run_id"]),
    (r"^/queries$", "query_comparison", []),
    (r"^/conclusions$", "record_conclusion", []),
    (r"^/conclusions/(?P<conclusion_id>[^/]+)/annotations$",
     "annotate", ["conclusion_id"]),
    (r"^/conclusions/(?P<conclusion_id>[^/]+)/validations$",
     "add_validation", ["conclusion_id"]),
]

ROUTES_GET = [
    (r"^/conclusions/(?P<conclusion_id>[^/]+)/trace$",
     "trace_conclusion", ["conclusion_id"]),
    (r"^/datasets/(?P<dataset_id>[^/]+)$", "get_dataset", ["dataset_id"]),
    (r"^/runs/(?P<run_id>[^/]+)$", "get_run", ["run_id"]),
    (r"^/versions/(?P<version_id>[^/]+)$", "get_version", ["version_id"]),
    (r"^/artifacts/affected$", "affected_artifacts", []),
    (r"^/events$", "events", []),
    (r"^/state$", "snapshot", []),
]


class LedgerService:
    """命令执行 + 持久化的薄事务层。

    服务级锁保证「校验—追加—落盘」对多线程 HTTP 请求串行执行；
    一次命令产生的多个事件（撤销级联、完成晋升）整体写入。
    """

    def __init__(self, ledger: Ledger, store: JsonlStore | None = None):
        self.ledger = ledger
        self.store = store
        self._tx_lock = threading.Lock()

    def command(self, method_name: str, *, args: list | None = None,
                kwargs: dict | None = None):
        args = args or []
        kwargs = kwargs or {}
        with self._tx_lock:
            before = len(self.ledger.events())
            result = getattr(self.ledger, method_name)(*args, **kwargs)
            if self.store is not None:
                self.store.append_many(self.ledger.events()[before:])
            return result

    def query(self, method_name: str, *, args: list | None = None,
              kwargs: dict | None = None):
        return getattr(self.ledger, method_name)(*(args or []), **(kwargs or {}))


def build_service(store_path: str | None = None) -> LedgerService:
    vocab = load_vocab()
    store = JsonlStore(store_path) if store_path else None
    if store is not None:
        ledger = replay(store.load_all(), vocab=vocab)
    else:
        ledger = Ledger(vocab=vocab)
    return LedgerService(ledger, store)


def _match(routes, path: str):
    for pattern, method_name, arg_names in routes:
        m = re.match(pattern, path)
        if m:
            return method_name, {name: m.group(name) for name in arg_names}
    return None


class Handler(BaseHTTPRequestHandler):
    service: LedgerService = None  # 由 make_server 注入到类属性

    def _send_json(self, status: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, code: str, message: str,
                    details: dict | None = None) -> None:
        self._send_json(status, {
            "error": {"code": code, "message": message, "details": details}})

    def _read_body(self) -> dict | None:
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._send_error(400, "BAD_JSON", "请求体不是合法 JSON")
            return None
        if not isinstance(body, dict):
            self._send_error(400, "BAD_REQUEST", "请求体必须是 JSON 对象")
            return None
        return body

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send_json(200, health())
            return
        matched = _match(ROUTES_GET, path)
        if matched is None:
            self._send_error(404, "NOT_FOUND", f"未知路径 {path}")
            return
        method_name, path_args = matched
        try:
            result = self.service.query(
                method_name, args=list(path_args.values()))
        except LedgerError as exc:
            self._send_ledger_error(exc)
            return
        self._send_json(200, {"data": result})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        matched = _match(ROUTES_POST, path)
        if matched is None:
            self._send_error(404, "NOT_FOUND", f"未知路径 {path}")
            return
        method_name, path_args = matched
        body = self._read_body()
        if body is None:
            return
        kwargs = dict(path_args)
        kwargs.update(body)
        try:
            result = self.service.command(method_name, kwargs=kwargs)
        except LedgerError as exc:
            self._send_ledger_error(exc)
            return
        except TypeError as exc:
            self._send_error(400, "BAD_REQUEST", f"参数不匹配：{exc}")
            return
        # 命令返回追加的事件（或幂等时的旧事件）
        self._send_json(201, {"data": result})

    def _send_ledger_error(self, exc: LedgerError) -> None:
        status = {
            "BAD_REQUEST": 400,
            "BAD_JSON": 400,
            "BAD_VOCAB": 400,
            "MAPPING_REQUIRED": 400,
            "MAPPING_MISMATCH": 422,
            "NOT_FOUND": 404,
            "CONFLICT": 409,
            "BAD_STATE": 409,
            "AUTH_DENIED": 403,
            "TAMPERED": 500,
        }.get(exc.code, 400)
        self._send_error(status, exc.code, exc.message, exc.details)

    def log_message(self, *_args):
        return


def make_server(port: int, store_path: str | None) -> ThreadingHTTPServer:
    service = build_service(store_path)

    class _Handler(Handler):
        pass

    _Handler.service = service
    return ThreadingHTTPServer(("0.0.0.0", port), _Handler)


def self_check(store_path: str | None = None) -> None:
    """配置自检：词表加载、审计事件重放与哈希链校验。"""
    vocab = load_vocab()
    required = ["数据许可", "运行状态", "证据等级", "运行类型", "质检结论"]
    missing = [k for k in required if k not in vocab]
    if missing:
        raise SystemExit(f"领域词表缺少分组：{missing}")
    if store_path:
        store = JsonlStore(store_path)
        count = len(store.load_all())
        replay(store.load_all(), vocab=vocab)
        print(f"基础检查通过：审计链重放成功（{store_path}，{count} 个事件）")
    else:
        print("基础检查通过")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="跨物种细胞图谱实验账本")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--check", action="store_true",
                        help="自检词表与审计哈希链后退出")
    parser.add_argument("--store", default="ledger.jsonl",
                        help="只追加审计事件文件（JSONL），留空用 '' 关闭持久化")
    args = parser.parse_args()
    store_path = args.store or None
    if args.check:
        self_check(store_path)
    else:
        make_server(args.port, store_path).serve_forever()
