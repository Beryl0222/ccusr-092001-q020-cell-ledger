"""服务身份与 HTTP 接口冒烟测试。"""

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from service import SERVICE_ID, build_handler, health
from ledger_core import Ledger


def _request(url: str, method="GET", payload=None):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class HealthTest(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(health(), {"status": "ok", "service": SERVICE_ID})


class HttpApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(Ledger(None)))
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_health(self):
        status, body = _request(f"{self.base}/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["service"], SERVICE_ID)

    def test_full_workflow_over_http(self):
        b = self.base
        # 注册受限数据集，未授权训练被拒（403）并留审计
        _, ds = _request(
            f"{b}/datasets", "POST",
            {"project_id": "owner", "species": "人类", "license": "项目内使用",
             "dataset_id": "ds-http"},
        )
        self.assertEqual(ds["dataset_id"], "ds-http")
        _request(f"{b}/datasets/ds-http/qc", "PUT", {"metrics": {"cells": 9}, "passed": True})
        _, model = _request(
            f"{b}/models", "POST",
            {"name": "m", "code_version": "c1", "model_id": "m-http"},
        )
        status, err = _request(
            f"{b}/runs", "POST",
            {"project_id": "other", "run_type": "训练", "dataset_ids": ["ds-http"],
             "model_id": "m-http", "code_version": "c1"},
        )
        self.assertEqual(status, 403)
        self.assertEqual(err["error"], "PermissionDeniedError")

        # 授权后跑通训练 → 完成 → 跨物种查询
        _request(f"{b}/grants", "POST",
                 {"dataset_id": "ds-http", "project_id": "other", "scope": "训练"})
        _, run = _request(
            f"{b}/runs", "POST",
            {"project_id": "other", "run_type": "训练", "dataset_ids": ["ds-http"],
             "model_id": "m-http", "code_version": "c1", "run_id": "run-http"},
        )
        self.assertEqual(run["status"], "排队")
        _request(f"{b}/runs/run-http/start", "PUT", {})
        _, version = _request(f"{b}/runs/run-http/complete", "PUT", {})
        self.assertEqual(version["status"], "正式版本")

        _, q = _request(
            f"{b}/queries", "POST",
            {"project_id": "other", "version_id": version["version_id"],
             "inputs": [{"species": "人类"}]},
        )
        _, concl = _request(
            f"{b}/conclusions", "POST",
            {"project_id": "other", "query_id": q["query_id"],
             "claim": "相似", "inputs": [{"cell": "c1"}], "conclusion_id": "concl-http"},
        )
        status, trace = _request(f"{b}/conclusions/concl-http/trace")
        self.assertEqual(status, 200)
        self.assertEqual(trace["version"]["code_version"], "c1")

        # 撤销许可：版本与结论级联标记
        _, report = _request(f"{b}/datasets/ds-http/revoke", "PUT", {"reason": "x"})
        self.assertIn(version["version_id"], report["affected_versions"])
        self.assertIn("concl-http", report["affected_conclusions"])

    def test_unknown_entity_returns_404_and_bad_json_400(self):
        status, _ = _request(f"{self.base}/conclusions/nope/trace")
        self.assertEqual(status, 404)
        status, body = _request(
            f"{self.base}/datasets", "POST", {"project_id": "p"}
        )
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
