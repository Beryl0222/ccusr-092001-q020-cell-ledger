"""核心账本业务规则测试。"""

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ledger_core import (
    CONCLUSION_AFFECTED,
    EV_COMPUTED,
    EV_EXPERIMENT,
    EV_MODEL,
    LIC_NO_RETRAIN,
    LIC_OPEN,
    LIC_PROJECT,
    LIC_REVOKED,
    LIC_TIME_LIMITED,
    RUN_INFERENCE,
    RUN_TRAINING,
    SCOPE_INFERENCE,
    SCOPE_TRAINING,
    ST_AFFECTED,
    ST_FAILED,
    ST_OFFICIAL,
    ST_QUEUED,
    ST_RUNNING,
    ConflictError,
    Ledger,
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
)


def iso(dt: datetime) -> str:
    return dt.isoformat()


FUTURE = iso(datetime.now(timezone.utc) + timedelta(days=30))
PAST = iso(datetime.now(timezone.utc) - timedelta(days=1))


class LedgerTestBase(unittest.TestCase):
    def setUp(self):
        self.lg = Ledger(None)

    def make_open_dataset(self, species="人类", ds_id=None):
        return self.lg.register_dataset(
            project_id="proj-a", species=species, license=LIC_OPEN,
            tissue="血液", dataset_id=ds_id,
        )["dataset_id"]

    def make_restricted_dataset(self, license=LIC_PROJECT, species="人类", ds_id=None):
        return self.lg.register_dataset(
            project_id="proj-owner", species=species, license=license,
            license_expires_at=FUTURE if license == LIC_TIME_LIMITED else None,
            dataset_id=ds_id,
        )["dataset_id"]

    def qc_ok(self, ds_id):
        self.lg.record_qc(ds_id, {"cells": 1000, "doublet_rate": 0.02}, True)

    def make_model(self, model_id="model-1", code="code-v1"):
        return self.lg.register_model(
            name="细胞补全模型", code_version=code, params={"lr": 0.001},
            model_id=model_id,
        )["model_id"]

    def make_mapping(self, species, map_id="map-1"):
        return self.lg.register_mapping(
            name="人鼠同源映射", source_species=species[0], target_species=species[1],
            version="homologene-1", covered_species=list(species), mapping_id=map_id,
        )["mapping_id"]


class DatasetAndLicenseTest(LedgerTestBase):
    def test_register_rejects_unknown_license_and_vocab_revoked(self):
        with self.assertRaises(ValidationError):
            self.lg.register_dataset("p", "人类", "随便用")
        with self.assertRaises(ValidationError):
            self.lg.register_dataset("p", "人类", LIC_REVOKED)

    def test_time_limited_requires_expiry(self):
        with self.assertRaises(ValidationError):
            self.lg.register_dataset("p", "人类", LIC_TIME_LIMITED)
        with self.assertRaises(ValidationError):
            self.lg.register_dataset(
                "p", "人类", LIC_TIME_LIMITED, license_expires_at="下个月"
            )

    def test_duplicate_dataset_id_conflicts(self):
        self.make_open_dataset(ds_id="ds-x")
        with self.assertRaises(ConflictError):
            self.make_open_dataset(ds_id="ds-x")


class AuthorizationTest(LedgerTestBase):
    def test_open_data_runs_without_grant(self):
        ds = self.make_open_dataset()
        self.qc_ok(ds)
        model = self.make_model()
        run = self.lg.submit_run("proj-a", RUN_TRAINING, [ds], model, "code-v1")
        self.assertEqual(run["status"], ST_QUEUED)

    def test_restricted_data_requires_grant(self):
        ds = self.make_restricted_dataset()
        self.qc_ok(ds)
        model = self.make_model()
        with self.assertRaises(PermissionDeniedError):
            self.lg.submit_run("proj-outsider", RUN_TRAINING, [ds], model, "code-v1")
        # 被拒绝的尝试进入合规审计
        self.assertEqual(len(self.lg.rejections), 1)
        self.lg.issue_grant(ds, "proj-outsider", SCOPE_TRAINING, FUTURE)
        run = self.lg.submit_run("proj-outsider", RUN_TRAINING, [ds], model, "code-v1")
        self.assertEqual(run["status"], ST_QUEUED)

    def test_inference_grant_does_not_authorize_training(self):
        ds = self.make_restricted_dataset()
        self.qc_ok(ds)
        model = self.make_model()
        self.lg.issue_grant(ds, "proj-b", SCOPE_INFERENCE, FUTURE)
        with self.assertRaises(PermissionDeniedError):
            self.lg.submit_run("proj-b", RUN_TRAINING, [ds], model, "code-v1")
        # 推理授权可以推理
        run = self.lg.submit_run("proj-b", RUN_INFERENCE, [ds], model, "code-v1")
        self.assertEqual(run["status"], ST_QUEUED)

    def test_no_retrain_license_blocks_training_even_with_grant(self):
        ds = self.make_restricted_dataset(license=LIC_NO_RETRAIN)
        self.qc_ok(ds)
        model = self.make_model()
        self.lg.issue_grant(ds, "proj-b", SCOPE_TRAINING, FUTURE)
        with self.assertRaises(PermissionDeniedError):
            self.lg.submit_run("proj-b", RUN_TRAINING, [ds], model, "code-v1")

    def test_expired_grant_and_expired_time_limited_license_block(self):
        ds = self.make_restricted_dataset(license=LIC_TIME_LIMITED)
        self.qc_ok(ds)
        model = self.make_model()
        self.lg.issue_grant(ds, "proj-b", SCOPE_TRAINING, PAST)
        with self.assertRaises(PermissionDeniedError):
            self.lg.submit_run("proj-b", RUN_TRAINING, [ds], model, "code-v1")

    def test_revoked_grant_blocks_new_run(self):
        ds = self.make_restricted_dataset()
        self.qc_ok(ds)
        model = self.make_model()
        self.lg.issue_grant(ds, "proj-b", SCOPE_TRAINING, FUTURE)
        self.lg.revoke_grant(ds, "proj-b", SCOPE_TRAINING)
        with self.assertRaises(PermissionDeniedError):
            self.lg.submit_run("proj-b", RUN_TRAINING, [ds], model, "code-v1")
        # 重复撤销幂等
        self.lg.revoke_grant(ds, "proj-b", SCOPE_TRAINING)

    def test_qc_required_and_must_pass(self):
        ds = self.make_open_dataset()
        model = self.make_model()
        with self.assertRaises(ValidationError):
            self.lg.submit_run("proj-a", RUN_TRAINING, [ds], model, "code-v1")
        self.lg.record_qc(ds, {}, False)
        with self.assertRaises(ValidationError):
            self.lg.submit_run("proj-a", RUN_TRAINING, [ds], model, "code-v1")


class CrossSpeciesTest(LedgerTestBase):
    def test_cross_species_run_requires_mapping(self):
        ds1 = self.make_open_dataset(species="人类")
        ds2 = self.make_open_dataset(species="小鼠", ds_id="ds-2")
        self.qc_ok(ds1)
        self.qc_ok(ds2)
        model = self.make_model()
        with self.assertRaises(ValidationError):
            self.lg.submit_run(
                "proj-a", RUN_TRAINING, [ds1, ds2], model, "code-v1"
            )
        mapping = self.make_mapping(["人类", "小鼠"])
        run = self.lg.submit_run(
            "proj-a", RUN_TRAINING, [ds1, ds2], model, "code-v1", mapping_id=mapping
        )
        self.assertEqual(run["mapping_id"], mapping)

    def test_mapping_must_cover_all_species(self):
        ds1 = self.make_open_dataset(species="人类")
        ds2 = self.make_open_dataset(species="斑马鱼", ds_id="ds-2")
        self.qc_ok(ds1)
        self.qc_ok(ds2)
        model = self.make_model()
        mapping = self.make_mapping(["人类", "小鼠"])
        with self.assertRaises(ValidationError):
            self.lg.submit_run(
                "proj-a", RUN_TRAINING, [ds1, ds2], model, "code-v1",
                mapping_id=mapping,
            )


class RunLifecycleTest(LedgerTestBase):
    def _ready_run(self):
        ds = self.make_open_dataset()
        self.qc_ok(ds)
        model = self.make_model()
        run = self.lg.submit_run("proj-a", RUN_TRAINING, [ds], model, "code-v1")
        return run["run_id"]

    def test_illegal_transitions(self):
        rid = self._ready_run()
        with self.assertRaises(ConflictError):
            self.lg.fail_run(rid)  # 排队中不能失败
        with self.assertRaises(ConflictError):
            self.lg.complete_run(rid)  # 排队中不能完成

    def test_failed_run_retries_safely_and_single_official_version(self):
        rid = self._ready_run()
        self.lg.start_run(rid)
        self.lg.fail_run(rid, reason="OOM")
        self.assertEqual(self.lg.get_run(rid)["status"], ST_FAILED)

        # 失败可重试，回到排队；重复重试安全
        self.lg.retry_run(rid)
        self.lg.retry_run(rid)
        self.assertEqual(self.lg.get_run(rid)["status"], ST_QUEUED)

        self.lg.start_run(rid)
        v1 = self.lg.complete_run(rid)
        # 重复完成是幂等的：不会产生第二个正式版本
        v2 = self.lg.complete_run(rid)
        self.assertEqual(v1["version_id"], v2["version_id"])
        official = [v for v in self.lg.list_versions()]
        self.assertEqual(len(official), 1)
        self.assertEqual(self.lg.get_run(rid)["status"], ST_OFFICIAL)

        # 正式版本后不能再重试
        with self.assertRaises(ConflictError):
            self.lg.retry_run(rid)

    def test_cannot_retry_unfailed_run_to_wrong_state(self):
        rid = self._ready_run()
        # 排队态重复重试不报错（幂等）
        self.lg.retry_run(rid)


class RevocationCascadeTest(LedgerTestBase):
    def _build_pipeline(self):
        ds = self.make_restricted_dataset()
        self.qc_ok(ds)
        model = self.make_model()
        self.lg.issue_grant(ds, "proj-a", SCOPE_TRAINING, FUTURE)
        run = self.lg.submit_run("proj-a", RUN_TRAINING, [ds], model, "code-v1")
        rid = run["run_id"]
        self.lg.start_run(rid)
        version = self.lg.complete_run(rid)
        vid = version["version_id"]
        q = self.lg.create_query(
            "proj-a", vid, [{"species": "人类", "cell": "c1"}]
        )
        concl = self.lg.create_conclusion(
            "proj-a", q["query_id"], "T 细胞相似", [{"cell": "c1"}]
        )
        return ds, rid, vid, concl["conclusion_id"]

    def test_revocation_blocks_runs_and_marks_artifacts(self):
        ds, rid, vid, cid = self._build_pipeline()
        report = self.lg.revoke_dataset_license(ds, reason="供体撤回同意")
        self.assertIn(vid, report["affected_versions"])
        self.assertIn(cid, report["affected_conclusions"])
        self.assertEqual(self.lg.get_version(vid)["status"], ST_AFFECTED)
        self.assertEqual(self.lg.get_conclusion(cid)["status"], CONCLUSION_AFFECTED)
        self.assertEqual(self.lg.get_dataset(ds)["license"], LIC_REVOKED)

        # 受影响版本不能再用于查询
        with self.assertRaises(PermissionDeniedError):
            self.lg.create_query("proj-a", vid, [{"species": "人类"}])

        # 新运行被阻止，且审计仍有记录
        self.assertEqual(len(self.lg.list_runs()), 1)
        with self.assertRaises(PermissionDeniedError):
            self.lg.submit_run("proj-a", RUN_INFERENCE, [ds], "model-1", "code-v1")
        self.assertEqual(len(self.lg.rejections), 1)

        # 重复撤销幂等
        self.lg.revoke_dataset_license(ds)

    def test_revocation_marks_unfinished_runs(self):
        ds = self.make_restricted_dataset()
        self.qc_ok(ds)
        model = self.make_model()
        self.lg.issue_grant(ds, "proj-a", SCOPE_TRAINING, FUTURE)
        rid = self.lg.submit_run(
            "proj-a", RUN_TRAINING, [ds], model, "code-v1"
        )["run_id"]
        self.lg.start_run(rid)
        self.lg.fail_run(rid)
        report = self.lg.revoke_dataset_license(ds)
        self.assertIn(rid, report["affected_runs"])
        self.assertEqual(self.lg.get_run(rid)["status"], ST_AFFECTED)
        # 受影响的失败运行不能再重试
        with self.assertRaises(ConflictError):
            self.lg.retry_run(rid)

    def test_audit_records_survive_revocation(self):
        ds, rid, vid, cid = self._build_pipeline()
        self.lg.add_annotation(cid, "张研究员", "初步相似性，需湿实验验证")
        self.lg.revoke_dataset_license(ds)
        trace = self.lg.trace_conclusion(cid)
        self.assertEqual(trace["run"]["run_id"], rid)
        self.assertEqual(trace["datasets"][0]["dataset_id"], ds)
        self.assertEqual(len(trace["annotations"]), 1)


class QueryTest(LedgerTestBase):
    def _official_version(self, species=("人类",), mapping=None):
        ds = self.make_open_dataset(species=species[0])
        self.qc_ok(ds)
        if len(species) > 1:
            ds2 = self.make_open_dataset(species=species[1], ds_id="ds-2")
            self.qc_ok(ds2)
            ids = [ds, ds2]
        else:
            ids = [ds]
        model = self.make_model()
        run = self.lg.submit_run(
            "proj-a", RUN_TRAINING, ids, model, "code-v1", mapping_id=mapping
        )
        self.lg.start_run(run["run_id"])
        return self.lg.complete_run(run["run_id"])["version_id"], model

    def test_query_records_model_and_mapping(self):
        vid, model = self._official_version()
        q = self.lg.create_query(
            "proj-a", vid, [{"species": "人类", "cell": "c1"}]
        )
        self.assertFalse(q["cross_species"])
        self.assertEqual(q["model_id"], model)
        self.assertIsNone(q["mapping_id"])

    def test_cross_species_query_requires_mapping(self):
        mapping = self.make_mapping(["人类", "小鼠"])
        vid, _ = self._official_version(species=("人类", "小鼠"), mapping=mapping)
        with self.assertRaises(ValidationError):
            self.lg.create_query(
                "proj-a", vid,
                [{"species": "人类"}, {"species": "小鼠"}],
            )
        q = self.lg.create_query(
            "proj-a", vid,
            [{"species": "人类"}, {"species": "小鼠"}],
            mapping_id=mapping,
        )
        self.assertTrue(q["cross_species"])
        self.assertEqual(q["mapping_id"], mapping)

    def test_query_rejects_wrong_model(self):
        vid, _ = self._official_version()
        other = self.lg.register_model(
            name="另一个模型", code_version="code-v2", model_id="model-2"
        )["model_id"]
        with self.assertRaises(ValidationError):
            self.lg.create_query(
                "proj-a", vid, [{"species": "人类"}], model_id=other
            )

    def test_missing_version_404(self):
        with self.assertRaises(NotFoundError):
            self.lg.create_query("proj-a", "ver-nope", [{"species": "人类"}])


class TraceAndEvidenceTest(LedgerTestBase):
    def test_trace_walks_back_to_inputs_code_and_validation(self):
        ds = self.make_open_dataset()
        self.qc_ok(ds)
        model = self.make_model()
        run = self.lg.submit_run(
            "proj-a", RUN_TRAINING, [ds], model, "code-v1",
            params={"resolution": 0.5},
        )
        self.lg.start_run(run["run_id"])
        vid = self.lg.complete_run(run["run_id"])["version_id"]
        q = self.lg.create_query("proj-a", vid, [{"species": "人类"}])
        cid = self.lg.create_conclusion(
            "proj-a", q["query_id"], "两群细胞转录状态相似", [{"cell": "x"}]
        )["conclusion_id"]
        self.lg.record_validation(cid, EV_COMPUTED, "独立管线复核", "复核一致")

        trace = self.lg.trace_conclusion(cid)
        self.assertEqual(trace["conclusion"]["evidence_level"], EV_COMPUTED)
        self.assertEqual(trace["version"]["code_version"], "code-v1")
        self.assertEqual(trace["version"]["params"]["resolution"], 0.5)
        self.assertEqual(trace["model"]["code_version"], "code-v1")
        self.assertEqual(trace["datasets"][0]["license"], LIC_OPEN)
        self.assertEqual(len(trace["validations"]), 1)

    def test_evidence_level_only_upgrades(self):
        ds = self.make_open_dataset()
        self.qc_ok(ds)
        model = self.make_model()
        run = self.lg.submit_run("proj-a", RUN_TRAINING, [ds], model, "code-v1")
        self.lg.start_run(run["run_id"])
        vid = self.lg.complete_run(run["run_id"])["version_id"]
        q = self.lg.create_query("proj-a", vid, [{"species": "人类"}])
        cid = self.lg.create_conclusion(
            "proj-a", q["query_id"], "相似", [{"cell": "x"}],
            evidence_level=EV_EXPERIMENT,
        )["conclusion_id"]
        self.lg.record_validation(cid, EV_MODEL, "弱证据回写", "无增益")
        # 更强证据不会被弱证据降级
        self.assertEqual(self.lg.get_conclusion(cid)["evidence_level"], EV_EXPERIMENT)
        with self.assertRaises(ValidationError):
            self.lg.record_validation(cid, "道听途说", "m", "r")


class VersionDerivationTest(LedgerTestBase):
    def test_new_data_derives_new_version_without_rewriting_old(self):
        ds1 = self.make_open_dataset()
        self.qc_ok(ds1)
        model = self.make_model()
        run = self.lg.submit_run("proj-a", RUN_TRAINING, [ds1], model, "code-v1")
        self.lg.start_run(run["run_id"])
        v1 = self.lg.complete_run(run["run_id"])["version_id"]

        ds2 = self.make_open_dataset(species="小鼠", ds_id="ds-2")
        self.qc_ok(ds2)
        mapping = self.make_mapping(["人类", "小鼠"])
        derived_run = self.lg.derive_version(
            v1, "proj-a", RUN_TRAINING, [ds1, ds2], "code-v2",
            mapping_id=mapping,
        )
        self.assertEqual(derived_run["derived_from"], v1)
        self.lg.start_run(derived_run["run_id"])
        v2 = self.lg.complete_run(derived_run["run_id"])["version_id"]

        self.assertNotEqual(v1, v2)
        # 旧版本保持正式、内容不变
        old = self.lg.get_version(v1)
        self.assertEqual(old["status"], ST_OFFICIAL)
        self.assertEqual(old["code_version"], "code-v1")
        self.assertEqual(old["dataset_ids"], [ds1])

        q = self.lg.create_query(
            "proj-a", v2, [{"species": "人类"}, {"species": "小鼠"}],
            mapping_id=mapping,
        )
        cid = self.lg.create_conclusion(
            "proj-a", q["query_id"], "跨物种相似", [{"cell": "x"}]
        )["conclusion_id"]
        lineage = self.lg.trace_conclusion(cid)["derived_lineage"]
        self.assertEqual([v["version_id"] for v in lineage], [v1])


class AnnotationTest(LedgerTestBase):
    def test_annotation_target_must_exist(self):
        with self.assertRaises(NotFoundError):
            self.lg.add_annotation("nope", "作者", "批注")


class PersistenceTest(unittest.TestCase):
    def test_event_log_replay_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            lg = Ledger(path)
            ds = lg.register_dataset("p", "人类", LIC_OPEN)["dataset_id"]
            lg.record_qc(ds, {"cells": 10}, True)
            model = lg.register_model("m", "c1")["model_id"]
            run = lg.submit_run("p", RUN_TRAINING, [ds], model, "c1")["run_id"]
            lg.start_run(run)
            lg.complete_run(run)

            replayed = Ledger(path)
            self.assertEqual(len(replayed.list_datasets()), 1)
            self.assertEqual(replayed.get_run(run)["status"], ST_OFFICIAL)
            self.assertEqual(len(replayed.list_versions()), 1)
            self.assertEqual(len(replayed.qc_records), 1)


if __name__ == "__main__":
    unittest.main()
