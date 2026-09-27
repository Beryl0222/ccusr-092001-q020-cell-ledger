"""服务端到端测试：HTTP 接口 + JSONL 持久化重启恢复。"""

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from service import SERVICE_ID, make_server


class ServerHarness:
    def __init__(self, store_path=None):
        self.httpd = make_server(0, store_path)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def request(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data, method=method,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())


def post(h, path, body):
    return h.request("POST", path, body)


def get(h, path):
    return h.request("GET", path)


def seed_full_flow(h):
    """项目→数据集→质检→授权→映射→代码→训练(失败→重试)→推理→查询→结论→验证。"""
    assert post(h, "/projects", {"project_id": "p1", "name": "联合图谱"})[0] == 201
    post(h, "/datasets", {
        "dataset_id": "d-human", "species": "人", "tissue": "外周血",
        "disease": "健康", "license": "开放研究", "gene_namespace": "HGNC",
        "sample_naming": "labA_<donor>", "source_uri": "s3://h",
        "checksum": "sha256:h"})
    post(h, "/datasets", {
        "dataset_id": "d-mouse", "species": "小鼠", "tissue": "脾",
        "disease": "炎症", "license": "项目内使用", "gene_namespace": "MGI",
        "sample_naming": "labB_m<id>", "source_uri": "s3://m",
        "checksum": "sha256:m"})
    post(h, "/datasets/d-human/qc", {"conclusion": "通过"})
    post(h, "/datasets/d-mouse/qc", {"conclusion": "有条件通过"})
    post(h, "/grants", {"grant_id": "g1", "dataset_id": "d-human",
                        "project_id": "p1"})
    post(h, "/grants", {"grant_id": "g2", "dataset_id": "d-mouse",
                        "project_id": "p1"})
    post(h, "/mappings", {
        "mapping_id": "map1", "name": "人鼠同源",
        "source": "Ensembl Compara 112",
        "species_pairs": [["人", "小鼠"]], "gene_count": 16800,
        "checksum": "sha256:map"})
    post(h, "/code-versions", {"code_version": "c1", "git_ref": "git:abc",
                               "image_digest": "sha256:img"})

    # 训练：先失败再重试，最终只有一个正式模型版本
    post(h, "/runs", {
        "run_id": "run-train", "project_id": "p1", "run_type": "训练",
        "dataset_ids": ["d-human"], "code_version": "c1",
        "image_digest": "sha256:img", "params": {"lr": 0.001}})
    post(h, "/runs/run-train/fail", {"reason": "OOM"})
    post(h, "/runs/run-train/retry", {})
    post(h, "/runs/run-train/complete", {
        "version_id": "model-1", "checksum": "sha256:model1",
        "metrics": {"loss": 0.1}})

    # 跨物种推理
    post(h, "/runs", {
        "run_id": "run-infer", "project_id": "p1", "run_type": "推理",
        "dataset_ids": ["d-mouse"], "code_version": "c1",
        "image_digest": "sha256:img", "params": {"k": 10},
        "mapping_id": "map1", "model_version_id": "model-1"})
    post(h, "/runs/run-infer/complete", {
        "version_id": "emb-1", "checksum": "sha256:emb1"})

    # 跨物种比较查询（显式声明映射）
    post(h, "/queries", {
        "query_id": "q1", "project_id": "p1",
        "query_dataset_id": "d-mouse", "embedding_version_id": "emb-1",
        "mapping_id": "map1", "params": {"metric": "cosine"}})
    post(h, "/conclusions", {
        "conclusion_id": "c1", "query_id": "q1",
        "hypothesis": "鼠脾炎症细胞与人外周血 T 细胞亚群相似",
        "cell_pair": {"query_cell": "m42", "reference_cell": "h7"}})
    post(h, "/conclusions/c1/annotations", {
        "researcher": "王研究员", "text": "建议流式验证"})
    post(h, "/conclusions/c1/validations", {
        "level": "实验验证", "experiment_ref": "facs://b3",
        "supports": True})


class ServiceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = str(Path(self.tmp.name) / "ledger.jsonl")
        self.h = ServerHarness(self.store)

    def tearDown(self):
        self.h.stop()
        self.tmp.cleanup()

    def test_health(self):
        status, body = get(self.h, "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ok", "service": SERVICE_ID})

    def test_full_flow_and_trace(self):
        seed_full_flow(self.h)
        status, body = get(self.h, "/conclusions/c1/trace")
        self.assertEqual(status, 200)
        trace = body["data"]
        self.assertEqual(
            trace["conclusion"]["current_evidence_level"], "实验验证")
        self.assertEqual(trace["query"]["mapping"]["mapping_id"], "map1")
        self.assertEqual(trace["query"]["mapping"]["source"],
                         "Ensembl Compara 112")
        self.assertEqual(trace["model"]["version_id"], "model-1")
        run = trace["embedding"]["produced_by_run"]
        self.assertEqual(run["code_version"], "c1")
        self.assertEqual(run["image_digest"], "sha256:img")
        # 嵌入运行一次成功；训练运行经历失败重试，两次尝试都留在谱系上
        self.assertEqual(
            [a["attempt_no"] for a in run["attempts"]], [1])
        self.assertEqual(
            [a["attempt_no"]
             for a in trace["model"]["produced_by_run"]["attempts"]],
            [1, 2])
        inputs = {i["dataset_id"]: i for i in trace["embedding"]["inputs"]}
        self.assertEqual(inputs["d-mouse"]["gene_namespace"], "MGI")
        self.assertEqual(
            inputs["d-mouse"]["authorization_at_run_start"]["grant_id"], "g2")
        self.assertEqual(trace["annotations"][0]["researcher"], "王研究员")

    def test_license_gate_returns_403(self):
        post(self.h, "/projects", {"project_id": "p1", "name": "x"})
        post(self.h, "/datasets", {
            "dataset_id": "d1", "species": "人", "tissue": "血",
            "disease": "健康", "license": "禁止再训练", "gene_namespace": "HGNC",
            "sample_naming": "x", "source_uri": "s3://x", "checksum": "c"})
        post(self.h, "/datasets/d1/qc", {"conclusion": "通过"})
        post(self.h, "/grants", {"grant_id": "g1", "dataset_id": "d1",
                                 "project_id": "p1"})
        post(self.h, "/code-versions", {"code_version": "c1", "git_ref": "g",
                                        "image_digest": "i"})
        status, body = post(self.h, "/runs", {
            "run_id": "r1", "project_id": "p1", "run_type": "训练",
            "dataset_ids": ["d1"], "code_version": "c1",
            "image_digest": "i", "params": {}})
        self.assertEqual(status, 403)
        self.assertEqual(body["error"]["code"], "AUTH_DENIED")
        self.assertTrue(body["error"]["details"]["violations"])

    def test_cross_species_without_mapping_returns_400(self):
        seed_full_flow(self.h)
        status, body = post(self.h, "/queries", {
            "query_id": "q-bad", "project_id": "p1",
            "query_dataset_id": "d-mouse", "embedding_version_id": "emb-1",
            "params": {}})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "MAPPING_REQUIRED")

    def test_bad_vocab_rejected(self):
        post(self.h, "/projects", {"project_id": "p1", "name": "x"})
        status, body = post(self.h, "/datasets", {
            "dataset_id": "d1", "species": "人", "tissue": "血",
            "disease": "健康", "license": "不存在的许可",
            "gene_namespace": "HGNC", "sample_naming": "x",
            "source_uri": "s3://x", "checksum": "c"})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "BAD_VOCAB")

    def test_revocation_flags_and_blocks(self):
        seed_full_flow(self.h)
        status, _ = post(self.h, "/grants/g2/revoke",
                         {"reason": "供应方退出"})
        self.assertEqual(status, 201)
        # 受影响嵌入不得再查询
        status, body = post(self.h, "/queries", {
            "query_id": "q2", "project_id": "p1",
            "query_dataset_id": "d-human", "embedding_version_id": "emb-1",
            "mapping_id": "map1", "params": {}})
        self.assertEqual(status, 403)
        # 受影响产物清单可见
        status, body = get(self.h, "/artifacts/affected")
        self.assertEqual(status, 200)
        flagged = {v["version_id"] for v in body["data"]["versions"]}
        self.assertIn("emb-1", flagged)
        # 审计事件仍全部保留
        status, body = get(self.h, "/events")
        kinds = {e["type"] for e in body["data"]}
        self.assertIn("AccessGranted", kinds)
        self.assertIn("AccessRevoked", kinds)
        self.assertIn("VersionFlagged", kinds)

    def test_complete_is_idempotent_single_version(self):
        seed_full_flow(self.h)
        # 重复回调完成同一运行
        status1, _ = post(self.h, "/runs/run-infer/complete", {
            "version_id": "emb-1", "checksum": "sha256:emb1"})
        self.assertEqual(status1, 201)
        status, body = get(self.h, "/versions/emb-1")
        self.assertEqual(status, 200)
        self.assertEqual(body["data"]["status"], "正式版本")
        status, body = get(self.h, "/state")
        formal = [v for v in body["data"]["versions"].values()
                  if v["status"] == "正式版本"]
        self.assertEqual({v["version_id"] for v in formal},
                         {"model-1", "emb-1"})

    def test_state_survives_restart_from_jsonl(self):
        seed_full_flow(self.h)
        post(self.h, "/grants/g2/revoke", {"reason": "供应方退出"})
        event_count = len(Path(self.store).read_text(encoding="utf-8")
                          .strip().splitlines())
        self.h.stop()

        # 用同一审计文件重启：哈希链重放，结论仍可完整追溯
        self.h = ServerHarness(self.store)
        status, body = get(self.h, "/conclusions/c1/trace")
        self.assertEqual(status, 200)
        self.assertEqual(
            body["data"]["conclusion"]["hypothesis"],
            "鼠脾炎症细胞与人外周血 T 细胞亚群相似")
        status, body = get(self.h, "/versions/emb-1")
        self.assertEqual(body["data"]["status"], "已受影响")
        status, body = get(self.h, "/events")
        self.assertEqual(len(body["data"]), event_count)

        # 重启后闸门仍然有效：撤销许可的数据不可开新运行
        status, body = post(self.h, "/runs", {
            "run_id": "r-after", "project_id": "p1", "run_type": "推理",
            "dataset_ids": ["d-mouse"], "code_version": "c1",
            "image_digest": "sha256:img", "params": {"k": 1},
            "mapping_id": "map1", "model_version_id": "model-1"})
        self.assertEqual(status, 403)


if __name__ == "__main__":
    unittest.main()
