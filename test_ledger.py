"""账本领域不变量测试。

覆盖：许可/质检闸门、撤销级联与审计保留、失败重试不产生双版本、
跨物种映射强制、反向追溯、派生新版本、证据等级、限期许可与防篡改重放。
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from ledger import (
    ART_AFFECTED,
    ART_FORMAL,
    LIC_NO_RETRAIN,
    LIC_OPEN,
    LIC_PROJECT,
    LIC_TERM,
    Ledger,
    LedgerError,
    ST_AFFECTED,
    ST_FAILED,
    ST_FORMAL,
    ST_RUNNING,
    TamperError,
    load_vocab,
    replay,
)
from store import JsonlStore


class Clock:
    """可控时钟：时间只向前推进，便于断言限期许可等时间规则。"""

    def __init__(self, start="2026-09-01T08:00:00Z"):
        self.now = start

    def __call__(self):
        return self.now

    def tick(self, minutes=1):
        date, time = self.now.split("T")
        hh, mm, ss = (int(x) for x in time.rstrip("Z").split(":"))
        total = hh * 60 + mm + minutes
        hh, mm = divmod(total, 60)
        self.now = f"{date}T{hh:02d}:{mm:02d}:{ss:02d}Z"
        return self.now


def make_ledger():
    return Ledger(vocab=load_vocab(), clock=Clock())


def seed_baseline(ledger: Ledger):
    """登记项目、人/鼠两个数据集、授权、质检与跨物种映射、代码版本。"""
    ledger.register_project("proj-a", "联合图谱 A 组", pi="陈博士")
    ledger.register_dataset(
        "ds-human", species="人", tissue="外周血", disease="健康",
        license=LIC_OPEN, gene_namespace="HGNC", sample_naming="labA_<donor>_<well>",
        source_uri="s3://atlas/human.h5ad", checksum="sha256:human0")
    ledger.register_dataset(
        "ds-mouse", species="小鼠", tissue="脾脏", disease="炎症",
        license=LIC_PROJECT, gene_namespace="MGI", sample_naming="labB_m<id>",
        source_uri="s3://atlas/mouse.h5ad", checksum="sha256:mouse0")
    ledger.record_qc("ds-human", "通过", evidence_uri="s3://qc/human.json")
    ledger.record_qc("ds-mouse", "有条件通过", notes="双细胞比例偏高",
                     evidence_uri="s3://qc/mouse.json")
    ledger.grant_access("g-human", "ds-human", "proj-a")
    ledger.grant_access("g-mouse", "ds-mouse", "proj-a")
    ledger.register_mapping(
        "map-hm", name="人鼠同源基因", source="Ensembl Compara 112",
        species_pairs=[["人", "小鼠"]], gene_count=16800,
        checksum="sha256:map0", dataset_ids=[])
    ledger.register_code_version("code-1", git_ref="git:abc123",
                                 image_digest="sha256:img1")
    return ledger


def train_and_complete(ledger, run_id="run-train", version="model-v1"):
    ledger.start_run(
        run_id, project_id="proj-a", run_type="训练",
        dataset_ids=["ds-human"], code_version="code-1",
        image_digest="sha256:img1", params={"lr": 0.001, "epochs": 50})
    ledger.complete_run(run_id, version, checksum="sha256:m1",
                        metrics={"loss": 0.12})
    return version


def infer_and_complete(ledger, run_id="run-infer", version="emb-e1"):
    ledger.start_run(
        run_id, project_id="proj-a", run_type="推理",
        dataset_ids=["ds-mouse"], code_version="code-1",
        image_digest="sha256:img1", params={"k": 10},
        mapping_id="map-hm", model_version_id="model-v1")
    ledger.complete_run(run_id, version, checksum="sha256:e1")
    return version


class GateTest(unittest.TestCase):
    def setUp(self):
        self.ledger = seed_baseline(make_ledger())

    def test_missing_qc_blocks_run(self):
        self.ledger.register_dataset(
            "ds-raw", species="人", tissue="肝", disease="健康",
            license=LIC_OPEN, gene_namespace="HGNC", sample_naming="x",
            source_uri="s3://x", checksum="sha256:x")
        self.ledger.grant_access("g-raw", "ds-raw", "proj-a")
        with self.assertRaises(LedgerError) as ctx:
            self.ledger.start_run(
                "r", project_id="proj-a", run_type="训练",
                dataset_ids=["ds-raw"], code_version="code-1",
                image_digest="sha256:img1", params={})
        self.assertEqual(ctx.exception.code, "AUTH_DENIED")
        self.assertIn("缺少质检记录", ctx.exception.details["violations"][0]["reason"])

    def test_failed_qc_blocks_run(self):
        self.ledger.record_qc("ds-human", "不通过")
        with self.assertRaises(LedgerError) as ctx:
            self.ledger.start_run(
                "r", project_id="proj-a", run_type="训练",
                dataset_ids=["ds-human"], code_version="code-1",
                image_digest="sha256:img1", params={})
        self.assertEqual(ctx.exception.code, "AUTH_DENIED")

    def test_no_grant_blocks_run(self):
        self.ledger.register_project("proj-b", "无授权组")
        with self.assertRaises(LedgerError) as ctx:
            self.ledger.start_run(
                "r", project_id="proj-b", run_type="训练",
                dataset_ids=["ds-human"], code_version="code-1",
                image_digest="sha256:img1", params={})
        self.assertEqual(ctx.exception.code, "AUTH_DENIED")
        self.assertIn("无有效授权", ctx.exception.details["violations"][0]["reason"])

    def test_no_retrain_license_blocks_training_only(self):
        self.ledger.register_dataset(
            "ds-restricted", species="人", tissue="脑", disease="阿尔茨海默",
            license=LIC_NO_RETRAIN, gene_namespace="HGNC", sample_naming="y",
            source_uri="s3://y", checksum="sha256:y")
        self.ledger.record_qc("ds-restricted", "通过")
        self.ledger.grant_access("g-r", "ds-restricted", "proj-a")
        # 训练被禁止
        with self.assertRaises(LedgerError) as ctx:
            self.ledger.start_run(
                "rt", project_id="proj-a", run_type="训练",
                dataset_ids=["ds-restricted"], code_version="code-1",
                image_digest="sha256:img1", params={})
        self.assertIn("禁止再训练", ctx.exception.details["violations"][0]["reason"])
        # 同一数据用于推理（先备好模型）允许
        train_and_complete(self.ledger)
        self.ledger.start_run(
            "ri", project_id="proj-a", run_type="推理",
            dataset_ids=["ds-restricted"], code_version="code-1",
            image_digest="sha256:img1", params={"k": 5},
            model_version_id="model-v1")

    def test_term_license_expiry_blocks_new_run(self):
        clock = self.ledger._clock
        self.ledger.register_dataset(
            "ds-term", species="人", tissue="皮肤", disease="健康",
            license=LIC_TERM, gene_namespace="HGNC", sample_naming="z",
            source_uri="s3://z", checksum="sha256:z",
            license_expires="2026-09-01T09:00:00Z")
        self.ledger.record_qc("ds-term", "通过")
        self.ledger.grant_access("g-term", "ds-term", "proj-a")
        self.ledger.start_run(
            "r-early", project_id="proj-a", run_type="训练",
            dataset_ids=["ds-term"], code_version="code-1",
            image_digest="sha256:img1", params={})
        self.ledger.fail_run("r-early", "节点宕机")
        clock.tick(120)  # 超过 09:00
        with self.assertRaises(LedgerError) as ctx:
            self.ledger.retry_run("r-early")
        self.assertEqual(ctx.exception.code, "AUTH_DENIED")


class RetryTest(unittest.TestCase):
    def setUp(self):
        self.ledger = seed_baseline(make_ledger())

    def test_failed_run_retries_yield_single_formal_version(self):
        self.ledger.start_run(
            "r1", project_id="proj-a", run_type="训练",
            dataset_ids=["ds-human"], code_version="code-1",
            image_digest="sha256:img1", params={"lr": 0.01})
        self.ledger.fail_run("r1", "OOM")
        self.assertEqual(self.ledger.runs["r1"]["status"], ST_FAILED)

        # 只有失败可重试的运行能重试
        with self.assertRaises(LedgerError):
            self.ledger.start_run(
                "r1", project_id="proj-a", run_type="训练",
                dataset_ids=["ds-human"], code_version="code-1",
                image_digest="sha256:img1", params={"lr": 0.01})
        self.ledger.retry_run("r1")
        self.assertEqual(self.ledger.runs["r1"]["status"], ST_RUNNING)
        self.assertEqual(
            [a["attempt_no"] for a in self.ledger.runs["r1"]["attempts"]], [1, 2])
        self.ledger.complete_run("r1", "model-v1", "sha256:m1")
        self.assertEqual(self.ledger.runs["r1"]["status"], ST_FORMAL)

        # 正式版本全局唯一
        formal = [v for v in self.ledger.versions.values()
                  if v["status"] == ART_FORMAL]
        self.assertEqual([v["version_id"] for v in formal], ["model-v1"])

        # 重复完成是幂等的，不产生第二版本
        self.ledger.complete_run("r1", "model-v1", "sha256:m1")
        self.assertEqual(len(self.ledger.versions), 1)

    def test_cannot_complete_or_retry_in_wrong_state(self):
        self.ledger.start_run(
            "r2", project_id="proj-a", run_type="训练",
            dataset_ids=["ds-human"], code_version="code-1",
            image_digest="sha256:img1", params={})
        with self.assertRaises(LedgerError):
            self.ledger.retry_run("r2")  # 运行中不可重试
        self.ledger.fail_run("r2", "断电")
        with self.assertRaises(LedgerError):
            self.ledger.complete_run("r2", "v", "sha256:v")  # 失败态不可完成


class CrossSpeciesTest(unittest.TestCase):
    def setUp(self):
        self.ledger = seed_baseline(make_ledger())
        train_and_complete(self.ledger)

    def test_cross_species_requires_mapping(self):
        with self.assertRaises(LedgerError) as ctx:
            self.ledger.start_run(
                "rx", project_id="proj-a", run_type="推理",
                dataset_ids=["ds-mouse"], code_version="code-1",
                image_digest="sha256:img1", params={"k": 10},
                model_version_id="model-v1")
        self.assertEqual(ctx.exception.code, "MAPPING_REQUIRED")

    def test_mapping_must_cover_species_pair(self):
        self.ledger.register_mapping(
            "map-hz", name="人斑马鱼", source="x",
            species_pairs=[["人", "斑马鱼"]], gene_count=10, checksum="c")
        with self.assertRaises(LedgerError) as ctx:
            self.ledger.start_run(
                "rx", project_id="proj-a", run_type="推理",
                dataset_ids=["ds-mouse"], code_version="code-1",
                image_digest="sha256:img1", params={"k": 10},
                mapping_id="map-hz", model_version_id="model-v1")
        self.assertEqual(ctx.exception.code, "MAPPING_MISMATCH")

    def test_query_records_exact_mapping_and_model(self):
        infer_and_complete(self.ledger)
        self.ledger.query_comparison(
            "q1", project_id="proj-a", query_dataset_id="ds-mouse",
            embedding_version_id="emb-e1", mapping_id="map-hm",
            params={"metric": "cosine"})
        q = self.ledger.queries["q1"]
        self.assertTrue(q["cross_species"])
        self.assertEqual(q["mapping"]["mapping_id"], "map-hm")
        self.assertEqual(q["mapping"]["source"], "Ensembl Compara 112")
        self.assertEqual(q["model_version_id"], "model-v1")

    def test_single_species_query_needs_no_mapping(self):
        # 人数据 + 人源模型 → 单物种嵌入；单物种查询无需映射
        self.ledger.start_run(
            "run-inf-h", project_id="proj-a", run_type="推理",
            dataset_ids=["ds-human"], code_version="code-1",
            image_digest="sha256:img1", params={"k": 10},
            model_version_id="model-v1")
        self.ledger.complete_run("run-inf-h", "emb-eh", "sha256:eh")
        self.ledger.query_comparison(
            "q-h", project_id="proj-a", query_dataset_id="ds-human",
            embedding_version_id="emb-eh", params={})
        self.assertFalse(self.ledger.queries["q-h"]["cross_species"])
        self.assertIsNone(self.ledger.queries["q-h"]["mapping"])


class RevocationTest(unittest.TestCase):
    def setUp(self):
        self.ledger = seed_baseline(make_ledger())
        train_and_complete(self.ledger)
        infer_and_complete(self.ledger)
        self.events_before = len(self.ledger.events())
        self.ledger.revoke_access("g-mouse", "供应方退出合作")

    def test_new_runs_blocked_after_revocation(self):
        with self.assertRaises(LedgerError):
            self.ledger.start_run(
                "rn", project_id="proj-a", run_type="推理",
                dataset_ids=["ds-mouse"], code_version="code-1",
                image_digest="sha256:img1", params={"k": 10},
                mapping_id="map-hm", model_version_id="model-v1")

    def test_queries_blocked_after_revocation(self):
        # 查询数据授权撤销、或参照嵌入输入授权撤销，均阻止查询
        with self.assertRaises(LedgerError) as ctx:
            self.ledger.query_comparison(
                "q2", project_id="proj-a", query_dataset_id="ds-mouse",
                embedding_version_id="emb-e1", mapping_id="map-hm", params={})
        self.assertEqual(ctx.exception.code, "AUTH_DENIED")
        with self.assertRaises(LedgerError) as ctx:
            self.ledger.query_comparison(
                "q3", project_id="proj-a", query_dataset_id="ds-human",
                embedding_version_id="emb-e1", mapping_id="map-hm", params={})
        self.assertEqual(ctx.exception.code, "AUTH_DENIED")

    def test_running_run_and_artifacts_flagged(self):
        # 撤销时刻正在运行的推理任务被级联标记，且不得晋升正式版本
        ledger2 = seed_baseline(make_ledger())
        train_and_complete(ledger2)
        infer_and_complete(ledger2)
        ledger2.start_run(
            "r-live2", project_id="proj-a", run_type="推理",
            dataset_ids=["ds-mouse"], code_version="code-1",
            image_digest="sha256:img1", params={"k": 10},
            mapping_id="map-hm", model_version_id="model-v1")
        ledger2.revoke_access("g-mouse", "供应方退出合作")
        self.assertEqual(ledger2.runs["r-live2"]["status"], ST_AFFECTED)
        with self.assertRaises(LedgerError) as ctx:
            ledger2.complete_run("r-live2", "emb-new", "sha256:x")
        self.assertEqual(ctx.exception.code, "AUTH_DENIED")

        # 撤销后尝试重试该（已被标记的）运行同样被拒绝
        # 先让其处于失败态不可能（已受影响不可 fail）；
        # 另起一个在撤销前失败的运行，撤销后重试被闸门挡住
        ledger3 = seed_baseline(make_ledger())
        train_and_complete(ledger3)
        ledger3.start_run(
            "r-wait", project_id="proj-a", run_type="推理",
            dataset_ids=["ds-mouse"], code_version="code-1",
            image_digest="sha256:img1", params={"k": 10},
            mapping_id="map-hm", model_version_id="model-v1")
        ledger3.fail_run("r-wait", "GPU 掉卡")
        ledger3.revoke_access("g-mouse", "供应方退出合作")
        with self.assertRaises(LedgerError) as ctx:
            ledger3.retry_run("r-wait")
        self.assertEqual(ctx.exception.code, "AUTH_DENIED")

    def test_embedding_and_model_lineage_flagged(self):
        # emb-e1 直接使用鼠数据 → 受影响；model-v1 仅用人数据，但 emb 是其下游
        self.assertEqual(self.ledger.versions["emb-e1"]["status"], ART_AFFECTED)
        affected = self.ledger.affected_artifacts()
        ids = {a["version_id"] for a in affected["versions"]}
        self.assertIn("emb-e1", ids)
        # 受影响产物不得再用于查询
        with self.assertRaises(LedgerError):
            self.ledger.query_comparison(
                "qx", project_id="proj-a", query_dataset_id="ds-human",
                embedding_version_id="emb-e1", params={})

    def test_audit_history_retained(self):
        # 撤销只追加事件，历史一条不少
        self.assertGreater(len(self.ledger.events()), self.events_before)
        grant = self.ledger.grants["g-mouse"]
        self.assertEqual(grant["status"], "已撤销")
        self.assertEqual(grant["revoke_reason"], "供应方退出合作")
        self.assertIsNotNone(grant["revoked_at"])
        # 原始授权事件仍在链上
        self.assertTrue(any(
            e["type"] == "AccessGranted" and e["payload"]["grant_id"] == "g-mouse"
            for e in self.ledger.events()))
        # 数据集与质检记录未被删除
        self.assertIn("ds-mouse", self.ledger.datasets)
        self.assertIn("ds-mouse", self.ledger.qc)

    def test_double_revoke_rejected(self):
        with self.assertRaises(LedgerError):
            self.ledger.revoke_access("g-mouse", "再次撤销")


class TraceAndDerivationTest(unittest.TestCase):
    def setUp(self):
        self.ledger = seed_baseline(make_ledger())
        train_and_complete(self.ledger)
        infer_and_complete(self.ledger)
        self.ledger.query_comparison(
            "q1", project_id="proj-a", query_dataset_id="ds-mouse",
            embedding_version_id="emb-e1", mapping_id="map-hm",
            params={"metric": "cosine", "k": 10})
        self.ledger.record_conclusion(
            "c1", query_id="q1",
            hypothesis="鼠脾脏炎症细胞与人外周血某 T 细胞亚群相似",
            cell_pair={"query_cell": "m_cell_42", "reference_cell": "h_cell_7"})
        self.ledger.annotate("c1", "王研究员", "建议用流式验证 CD3/CD8 标记")

    def test_trace_walks_back_to_inputs_code_and_evidence(self):
        trace = self.ledger.trace_conclusion("c1")
        self.assertEqual(trace["conclusion"]["conclusion_id"], "c1")
        self.assertEqual(trace["conclusion"]["current_evidence_level"], "模型推断")
        # 查询明确记录映射与模型
        self.assertEqual(trace["query"]["mapping"]["mapping_id"], "map-hm")
        self.assertEqual(trace["model"]["version_id"], "model-v1")
        # 嵌入 → 运行 → 输入 → 质检/许可快照/基因命名空间/校验和
        emb_run = trace["embedding"]["produced_by_run"]
        self.assertEqual(emb_run["code_version"], "code-1")
        self.assertEqual(emb_run["image_digest"], "sha256:img1")
        inputs = {i["dataset_id"]: i for i in trace["embedding"]["inputs"]}
        self.assertEqual(inputs["ds-mouse"]["gene_namespace"], "MGI")
        self.assertEqual(inputs["ds-mouse"]["qc"]["conclusion"], "有条件通过")
        self.assertEqual(
            inputs["ds-mouse"]["authorization_at_run_start"]["grant_id"],
            "g-mouse")
        self.assertEqual(emb_run["params"], {"k": 10})
        # 批注可追溯
        self.assertEqual(trace["annotations"][0]["researcher"], "王研究员")

    def test_validation_updates_evidence_level(self):
        self.assertEqual(self.ledger.current_evidence_level("c1"), "模型推断")
        self.ledger.add_validation(
            "c1", level="计算复核", experiment_ref="notebook://rerun-9",
            supports=True)
        self.assertEqual(self.ledger.current_evidence_level("c1"), "计算复核")
        self.ledger.add_validation(
            "c1", level="实验验证", experiment_ref="elisa://batch-2",
            supports=False, notes="未能重复")
        # 否定证据不提升等级
        self.assertEqual(self.ledger.current_evidence_level("c1"), "计算复核")
        self.ledger.add_validation(
            "c1", level="实验验证", experiment_ref="facs://batch-3",
            supports=True)
        self.assertEqual(self.ledger.current_evidence_level("c1"), "实验验证")

    def test_derive_new_version_keeps_old_immutable(self):
        # 新数据到来：派生 model-v2，旧 model-v1 与 emb-e1 保持不变
        self.ledger.register_dataset(
            "ds-human2", species="人", tissue="骨髓", disease="白血病",
            license=LIC_OPEN, gene_namespace="HGNC", sample_naming="labC_<id>",
            source_uri="s3://atlas/human2.h5ad", checksum="sha256:human2")
        self.ledger.record_qc("ds-human2", "通过")
        self.ledger.grant_access("g-human2", "ds-human2", "proj-a")
        self.ledger.start_run(
            "run-v2", project_id="proj-a", run_type="训练",
            dataset_ids=["ds-human2"], code_version="code-1",
            image_digest="sha256:img1", params={"lr": 0.0005},
            derived_from="model-v1")
        self.ledger.complete_run("run-v2", "model-v2", "sha256:m2")
        v1 = self.ledger.versions["model-v1"]
        self.assertEqual(v1["status"], ART_FORMAL)
        self.assertEqual(v1["checksum"], "sha256:m1")
        v2 = self.ledger.versions["model-v2"]
        self.assertEqual(v2["derived_from"], "model-v1")
        # 旧结论仍追溯到旧版本，不被改写
        trace = self.ledger.trace_conclusion("c1")
        self.assertEqual(trace["model"]["version_id"], "model-v1")
        # v2 的谱系里能看到父版本
        self.assertEqual(
            self.ledger.get_version("model-v2")["derived_from"], "model-v1")


class PersistenceTest(unittest.TestCase):
    def test_jsonl_roundtrip_replays_identical_state(self):
        ledger = seed_baseline(make_ledger())
        train_and_complete(ledger)
        infer_and_complete(ledger)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            store = JsonlStore(path)
            store.append_many(ledger.events())
            replayed = replay(store.load_all(), vocab=load_vocab())
            self.assertEqual(
                json.dumps(replayed.snapshot(), ensure_ascii=False,
                           sort_keys=True),
                json.dumps(ledger.snapshot(), ensure_ascii=False,
                           sort_keys=True))
            self.assertEqual(
                [e["hash"] for e in replayed.events()],
                [e["hash"] for e in ledger.events()])

    def test_tampered_event_chain_rejected(self):
        ledger = seed_baseline(make_ledger())
        train_and_complete(ledger)
        events = copy.deepcopy(ledger.events())
        # 篡改中间事件的负载（改数据集校验和）
        self.assertEqual(events[2]["type"], "DatasetRegistered")
        events[2]["payload"]["checksum"] = "sha256:forged"
        with self.assertRaises(TamperError):
            replay(events, vocab=load_vocab())
        # 删除一个事件（断链）同样被发现
        with self.assertRaises(TamperError):
            replay(ledger.events()[1:], vocab=load_vocab())


if __name__ == "__main__":
    unittest.main()
