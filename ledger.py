"""跨物种细胞图谱实验账本的领域核心。

所有状态变更都以只追加事件（Event）记录，事件之间以 sha256 哈希链相连。
当前状态由重放事件得到：许可撤销、产物标记等只会追加新事件，绝不改写或
删除历史记录，从而满足合规审计与反向追溯。

典型用法::

    ledger = Ledger(load_vocab(), clock=lambda: "2026-09-27T10:00:00Z")
    ledger.register_project(...)
    ledger.register_dataset(...)
    ledger.grant_access(...)
    ledger.start_run(...)
    ledger.complete_run(...)

事件可通过 :meth:`Ledger.events` 导出为 JSONL，再用 :func:`replay` 重建。
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

VOCAB_PATH = Path(__file__).with_name("domain.json")

# 运行类型
RUN_TRAINING = "训练"
RUN_INFERENCE = "推理"

# 运行状态（与 domain.json 词表一致）
ST_PENDING = "待校验"
ST_QUEUED = "排队"
ST_RUNNING = "运行中"
ST_FAILED = "失败可重试"
ST_FORMAL = "正式版本"
ST_AFFECTED = "已受影响"

# 许可
LIC_OPEN = "开放研究"
LIC_PROJECT = "项目内使用"
LIC_NO_RETRAIN = "禁止再训练"
LIC_TERM = "限期使用"
LIC_REVOKED = "已撤销"

# 授权 / 质检 / 产物状态
GRANT_VALID = "有效"
GRANT_REVOKED = "已撤销"

QC_PASS = "通过"
QC_FAIL = "不通过"
QC_CONDITIONAL = "有条件通过"

ART_CANDIDATE = "候选"
ART_FORMAL = "正式版本"
ART_AFFECTED = "已受影响"

# 产物种类
ART_MODEL = "模型版本"
ART_EMBEDDING = "嵌入版本"


def load_vocab(path: str | Path | None = None) -> dict:
    """读取领域词表（枚举值的唯一权威来源）。"""
    with open(path or VOCAB_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def utc_now() -> str:
    """默认时钟：UTC ISO-8601（秒级）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class LedgerError(Exception):
    """违反账本规则。``code`` 供 API 层映射为稳定的错误码。"""

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class TamperError(LedgerError):
    """重放时哈希链校验失败：审计记录被改动过。"""


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def event_hash(prev_hash: str, ts: str, actor: str, etype: str, payload: dict) -> str:
    h = hashlib.sha256()
    h.update(prev_hash.encode())
    h.update(ts.encode())
    h.update(actor.encode())
    h.update(etype.encode())
    h.update(_canonical(payload))
    return h.hexdigest()


def _parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


class Ledger:
    """只追加实验账本。

    命令方法在追加事件前先做闸门校验（质检、许可、授权、映射、版本状态），
    校验失败抛 :class:`LedgerError`，账本保持不变。
    """

    def __init__(self, vocab: dict | None = None, clock=utc_now):
        self.vocab = vocab or load_vocab()
        self._clock = clock
        self._lock = threading.RLock()
        self._events: list[dict] = []
        self._prev_hash = ""
        self._reset_state()

    # ------------------------------------------------------------------ 状态

    def _reset_state(self) -> None:
        self.projects: dict[str, dict] = {}
        self.datasets: dict[str, dict] = {}
        self.qc: dict[str, dict] = {}
        self.grants: dict[str, dict] = {}
        # dataset_id -> [grant_id...]
        self.dataset_grants: dict[str, list[str]] = defaultdict(list)
        self.mappings: dict[str, dict] = {}
        self.code_versions: dict[str, dict] = {}
        self.runs: dict[str, dict] = {}
        # 产物（模型 / 嵌入）版本：version_id -> record
        self.versions: dict[str, dict] = {}
        self.queries: dict[str, dict] = {}
        self.conclusions: dict[str, dict] = {}
        self.annotations: dict[str, list[dict]] = defaultdict(list)
        self.validations: dict[str, list[dict]] = defaultdict(list)
        # 幂等：run_id -> 完成时产生的正式版本 id
        self.run_versions: dict[str, str] = {}

    # ------------------------------------------------------------ 事件追加

    def _append(self, etype: str, payload: dict, actor: str = "平台") -> dict:
        """追加事件并折叠到当前状态。调用方必须持有锁或已完成全部校验。"""
        ts = self._clock()
        h = event_hash(self._prev_hash, ts, actor, etype, payload)
        event = {
            "seq": len(self._events) + 1,
            "ts": ts,
            "actor": actor,
            "type": etype,
            "payload": copy.deepcopy(payload),
            "prev_hash": self._prev_hash,
            "hash": h,
        }
        self._events.append(event)
        self._prev_hash = h
        self._fold(event)
        return event

    def _fold(self, event: dict) -> None:
        """把单个事件折叠进物化状态（重放与实时追加共用同一逻辑）。"""
        p = event["payload"]
        t = event["type"]
        if t == "ProjectRegistered":
            self.projects[p["project_id"]] = dict(p, registered_at=event["ts"])
        elif t == "DatasetRegistered":
            self.datasets[p["dataset_id"]] = dict(p, registered_at=event["ts"])
        elif t == "QcRecorded":
            # 质检只保留最新结论，但历史事件仍在审计链中
            self.qc[p["dataset_id"]] = dict(p, recorded_at=event["ts"])
        elif t == "AccessGranted":
            grant = dict(p, status=GRANT_VALID, granted_at=event["ts"],
                         revoked_at=None, revoke_reason=None)
            self.grants[p["grant_id"]] = grant
            self.dataset_grants[p["dataset_id"]].append(p["grant_id"])
        elif t == "AccessRevoked":
            grant = self.grants[p["grant_id"]]
            grant["status"] = GRANT_REVOKED
            grant["revoked_at"] = event["ts"]
            grant["revoke_reason"] = p["reason"]
        elif t == "MappingRegistered":
            self.mappings[p["mapping_id"]] = dict(
                p, status="可用", registered_at=event["ts"])
        elif t == "CodeVersionRegistered":
            self.code_versions[p["code_version"]] = dict(p, registered_at=event["ts"])
        elif t == "RunStarted":
            run = self.runs.get(p["run_id"])
            if run is None:
                run = {
                    "run_id": p["run_id"],
                    "project_id": p["project_id"],
                    "run_type": p["run_type"],
                    "status": ST_QUEUED,
                    "dataset_ids": list(p["inputs"]),
                    "model_version_id": p.get("model_version_id"),
                    "mapping_id": p.get("mapping_id"),
                    "params": dict(p["params"]),
                    "code_version": p["code_version"],
                    "image_digest": p["image_digest"],
                    "derived_from": p.get("derived_from"),
                    "license_snapshot": dict(p["license_snapshot"]),
                    "attempts": [],
                }
                self.runs[p["run_id"]] = run
            run["attempts"].append({
                "attempt_no": p["attempt_no"],
                "started_at": event["ts"],
                "actor": event["actor"],
            })
            run["status"] = ST_RUNNING
        elif t == "RunFailed":
            run = self.runs[p["run_id"]]
            run["status"] = ST_FAILED
            run["last_error"] = p["reason"]
            run["attempts"][-1]["failed_at"] = event["ts"]
            run["attempts"][-1]["error"] = p["reason"]
        elif t == "RunCompleted":
            run = self.runs[p["run_id"]]
            run["status"] = ST_FORMAL
            run.pop("last_error", None)
            run["attempts"][-1]["completed_at"] = event["ts"]
            run["attempts"][-1]["version_id"] = p["version_id"]
        elif t == "RunFlagged":
            self.runs[p["run_id"]]["status"] = ST_AFFECTED
            self.runs[p["run_id"]]["flag_reason"] = p["reason"]
        elif t == "VersionRegistered":
            self.versions[p["version_id"]] = dict(
                p, status=ART_CANDIDATE, registered_at=event["ts"])
        elif t == "VersionPromoted":
            self.versions[p["version_id"]]["status"] = ART_FORMAL
            self.versions[p["version_id"]]["promoted_at"] = event["ts"]
            self.run_versions[p["run_id"]] = p["version_id"]
        elif t == "VersionFlagged":
            self.versions[p["version_id"]]["status"] = ART_AFFECTED
            self.versions[p["version_id"]]["flag_reason"] = p["reason"]
            self.versions[p["version_id"]]["flagged_at"] = event["ts"]
        elif t == "MappingFlagged":
            self.mappings[p["mapping_id"]]["status"] = ART_AFFECTED
            self.mappings[p["mapping_id"]]["flag_reason"] = p["reason"]
        elif t == "ComparisonQueried":
            self.queries[p["query_id"]] = dict(p, queried_at=event["ts"])
        elif t == "ConclusionRecorded":
            self.conclusions[p["conclusion_id"]] = dict(
                p, recorded_at=event["ts"])
        elif t == "AnnotationAdded":
            self.annotations[p["conclusion_id"]].append(
                dict(p, annotated_at=event["ts"]))
        elif t == "ValidationRecorded":
            self.validations[p["conclusion_id"]].append(
                dict(p, validated_at=event["ts"]))
        else:  # pragma: no cover - 新增事件类型必须在此处理
            raise LedgerError("UNKNOWN_EVENT", f"未知事件类型 {t}")

    # ------------------------------------------------------------- 基础校验

    @staticmethod
    def _require_id(value, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise LedgerError("BAD_REQUEST", f"{name} 不能为空")
        return value.strip()

    def _require_vocab(self, group: str, value: str) -> str:
        allowed = self.vocab.get(group, [])
        if value not in allowed:
            raise LedgerError(
                "BAD_VOCAB", f"{value} 不在词表 {group} 中",
                {"field": group, "value": value, "allowed": allowed})
        return value

    def _require_existing(self, store: dict, key: str, kind: str):
        if key not in store:
            raise LedgerError("NOT_FOUND", f"{kind} {key} 不存在", {"id": key})
        return store[key]

    # --------------------------------------------------------- 接入与授权

    def register_project(self, project_id: str, name: str, pi: str = "",
                         actor: str = "平台") -> dict:
        with self._lock:
            pid = self._require_id(project_id, "project_id")
            self._require_id(name, "name")
            if pid in self.projects:
                raise LedgerError("CONFLICT", f"项目 {pid} 已登记")
            return self._append("ProjectRegistered", {
                "project_id": pid, "name": name, "pi": pi}, actor)

    def register_dataset(self, dataset_id: str, *, species: str, tissue: str,
                         disease: str, license: str, gene_namespace: str,
                         sample_naming: str, source_uri: str, checksum: str,
                         license_expires: str | None = None,
                         actor: str = "平台") -> dict:
        """登记数据集接入。许可必须取自词表；基因标识体系与样本命名必须显式记录。"""
        with self._lock:
            did = self._require_id(dataset_id, "dataset_id")
            if did in self.datasets:
                raise LedgerError("CONFLICT", f"数据集 {did} 已登记")
            for field, value in (("species", species), ("tissue", tissue),
                                 ("gene_namespace", gene_namespace),
                                 ("sample_naming", sample_naming),
                                 ("source_uri", source_uri),
                                 ("checksum", checksum)):
                self._require_id(value, field)
            self._require_vocab("数据许可", license)
            if license == LIC_TERM and not license_expires:
                raise LedgerError(
                    "BAD_REQUEST", "限期使用许可必须提供 license_expires")
            return self._append("DatasetRegistered", {
                "dataset_id": did,
                "species": species,
                "tissue": tissue,
                "disease_state": disease,
                "license": license,
                "gene_namespace": gene_namespace,
                "sample_naming": sample_naming,
                "source_uri": source_uri,
                "checksum": checksum,
                "license_expires": license_expires,
            }, actor)

    def record_qc(self, dataset_id: str, conclusion: str, notes: str = "",
                  evidence_uri: str = "", actor: str = "平台") -> dict:
        """记录质检结论。未通过质检的数据集不能进入任何运行。"""
        with self._lock:
            self._require_existing(self.datasets, dataset_id, "数据集")
            self._require_vocab("质检结论", conclusion)
            return self._append("QcRecorded", {
                "dataset_id": dataset_id,
                "conclusion": conclusion,
                "notes": notes,
                "evidence_uri": evidence_uri,
            }, actor)

    def grant_access(self, grant_id: str, dataset_id: str, project_id: str,
                     scope: str = "全部", actor: str = "平台") -> dict:
        """登记项目对数据集的使用授权。每个授权都可被单独撤销。"""
        with self._lock:
            gid = self._require_id(grant_id, "grant_id")
            if gid in self.grants:
                raise LedgerError("CONFLICT", f"授权 {gid} 已存在")
            self._require_existing(self.datasets, dataset_id, "数据集")
            self._require_existing(self.projects, project_id, "项目")
            return self._append("AccessGranted", {
                "grant_id": gid,
                "dataset_id": dataset_id,
                "project_id": project_id,
                "scope": scope,
            }, actor)

    def revoke_access(self, grant_id: str, reason: str, actor: str = "管理员") -> dict:
        """撤销授权并级联处置（单条命令、原子追加）：

        1. 授权置为已撤销——此后任何新运行 / 查询闸门失败；
        2. 使用该数据集的排队/运行中运行标记为已受影响，不得晋升正式版本；
        3. 相关产物（模型/嵌入版本）与映射标记为已受影响；
        4. 审计事件、运行记录与谱系全部保留，不删除任何数据。
        """
        with self._lock:
            grant = self._require_existing(self.grants, grant_id, "授权")
            if grant["status"] == GRANT_REVOKED:
                raise LedgerError("CONFLICT", f"授权 {grant_id} 已撤销")
            self._require_id(reason, "reason")
            dataset_id = grant["dataset_id"]
            project_id = grant["project_id"]
            events = [self._append("AccessRevoked", {
                "grant_id": grant_id,
                "dataset_id": dataset_id,
                "project_id": project_id,
                "reason": reason,
            }, actor)]

            # 受影响运行：仍在排队/运行中（已失败的运行本就不能晋升，无需标记）
            active_states = {ST_QUEUED, ST_RUNNING}
            for run in self.runs.values():
                if (run["project_id"] == project_id
                        and dataset_id in run["dataset_ids"]
                        and run["status"] in active_states):
                    events.append(self._append("RunFlagged", {
                        "run_id": run["run_id"],
                        "grant_id": grant_id,
                        "dataset_id": dataset_id,
                        "reason": f"输入数据授权已撤销：{reason}",
                    }, actor))

            # 受影响产物：由使用该数据集的运行产生的模型/嵌入版本，
            # 并沿「模型 → 下游嵌入」谱系传递性扩散
            flagged_ids: set[str] = set()
            for version in list(self.versions.values()):
                rid = version["run_id"]
                run = self.runs.get(rid)
                if (run and run["project_id"] == project_id
                        and dataset_id in run["dataset_ids"]
                        and version["status"] != ART_AFFECTED):
                    self._flag_version(version["version_id"], grant_id,
                                       dataset_id, reason, actor, events)
                    flagged_ids.add(version["version_id"])

            # 不动点：基于受影响模型的嵌入、以及产出受影响嵌入的运行，
            # 全部逐级标记（模型污染向下游推理结果传播）
            changed = True
            while changed:
                changed = False
                for version in list(self.versions.values()):
                    if version["status"] == ART_AFFECTED:
                        continue
                    tainted = (version["kind"] == ART_EMBEDDING
                               and version.get("model_version_id") in flagged_ids)
                    if tainted:
                        self._flag_version(version["version_id"], grant_id,
                                           dataset_id, reason, actor, events,
                                           prefix="上游模型训练数据授权已撤销")
                        flagged_ids.add(version["version_id"])
                        changed = True
                for run in self.runs.values():
                    if (run["status"] in active_states
                            and run.get("model_version_id") in flagged_ids):
                        events.append(self._append("RunFlagged", {
                            "run_id": run["run_id"],
                            "grant_id": grant_id,
                            "dataset_id": dataset_id,
                            "reason": "上游模型训练数据授权已撤销，推理运行受影响",
                        }, actor))
                        changed = True

            # 为该数据集构建的跨物种映射同样受影响（基因同源依据失效）
            for mapping in self.mappings.values():
                if (dataset_id in mapping.get("dataset_ids", [])
                        and mapping["status"] != ART_AFFECTED):
                    events.append(self._append("MappingFlagged", {
                        "mapping_id": mapping["mapping_id"],
                        "grant_id": grant_id,
                        "dataset_id": dataset_id,
                        "reason": f"映射依据的数据授权已撤销：{reason}",
                    }, actor))
            return events[0]

    def _flag_version(self, version_id: str, grant_id: str, dataset_id: str,
                      reason: str, actor: str, events: list,
                      prefix: str = "训练/输入数据授权已撤销") -> None:
        events.append(self._append("VersionFlagged", {
            "version_id": version_id,
            "grant_id": grant_id,
            "dataset_id": dataset_id,
            "reason": f"{prefix}：{reason}",
        }, actor))

    def register_mapping(self, mapping_id: str, *, name: str, source: str,
                         species_pairs: list[list[str]], gene_count: int,
                         checksum: str, dataset_ids: list[str] | None = None,
                         actor: str = "平台") -> dict:
        """登记跨物种基因标识映射（如同源基因库版本）。

        ``species_pairs`` 显式声明映射覆盖的物种对，``source`` 记录来源
        数据库与版本（如 Ensembl Compara 112），使每条跨物种结论都能说清
        用了哪一套映射。
        """
        with self._lock:
            mid = self._require_id(mapping_id, "mapping_id")
            if mid in self.mappings:
                raise LedgerError("CONFLICT", f"映射 {mid} 已登记")
            pairs = [tuple(sorted(p)) for p in species_pairs]
            if len(pairs) != len(set(pairs)):
                raise LedgerError("BAD_REQUEST", "species_pairs 存在重复物种对")
            for did in dataset_ids or []:
                self._require_existing(self.datasets, did, "数据集")
            return self._append("MappingRegistered", {
                "mapping_id": mid,
                "name": name,
                "source": source,
                "species_pairs": [list(p) for p in pairs],
                "gene_count": gene_count,
                "checksum": checksum,
                "dataset_ids": list(dataset_ids or []),
            }, actor)

    def register_code_version(self, code_version: str, git_ref: str,
                              image_digest: str, actor: str = "平台") -> dict:
        with self._lock:
            cv = self._require_id(code_version, "code_version")
            if cv in self.code_versions:
                raise LedgerError("CONFLICT", f"代码版本 {cv} 已登记")
            return self._append("CodeVersionRegistered", {
                "code_version": cv,
                "git_ref": git_ref,
                "image_digest": image_digest,
            }, actor)

    # ------------------------------------------------------------- 闸门逻辑

    def _valid_grant(self, dataset_id: str, project_id: str, at: str) -> dict | None:
        """返回数据集在某时刻对项目有效的授权快照，否则 None。"""
        ds = self.datasets[dataset_id]
        if ds["license"] == LIC_REVOKED:
            return None
        for gid in self.dataset_grants.get(dataset_id, []):
            g = self.grants[gid]
            if g["project_id"] != project_id or g["status"] != GRANT_VALID:
                continue
            if g["granted_at"] > at:
                continue
            if ds["license"] == LIC_TERM:
                expires = ds["license_expires"]
                if expires and at > expires:
                    continue
            return g
        return None

    def _mapping_covers(self, mapping: dict, species: set[str]) -> bool:
        pairs = {tuple(p) for p in mapping["species_pairs"]}
        species = sorted(species)
        for i, a in enumerate(species):
            for b in species[i + 1:]:
                if tuple(sorted((a, b))) not in pairs:
                    return False
        return True

    def _authorize_inputs(self, dataset_ids: list[str], project_id: str,
                          run_type: str, at: str) -> dict:
        """对一组输入数据集执行质检 + 许可 + 授权闸门，返回许可快照。"""
        if not dataset_ids:
            raise LedgerError("BAD_REQUEST", "运行至少需要一个输入数据集")
        if len(dataset_ids) != len(set(dataset_ids)):
            raise LedgerError("BAD_REQUEST", "输入数据集存在重复")
        snapshot = {}
        violations = []
        for did in dataset_ids:
            ds = self._require_existing(self.datasets, did, "数据集")
            qc = self.qc.get(did)
            if qc is None:
                violations.append({"dataset_id": did, "reason": "缺少质检记录"})
                continue
            if qc["conclusion"] == QC_FAIL:
                violations.append({"dataset_id": did, "reason": "质检未通过"})
                continue
            if ds["license"] == LIC_REVOKED:
                violations.append({"dataset_id": did, "reason": "数据集许可已撤销"})
                continue
            grant = self._valid_grant(did, project_id, at)
            if grant is None:
                violations.append({
                    "dataset_id": did,
                    "reason": f"项目 {project_id} 对该数据集无有效授权",
                })
                continue
            if run_type == RUN_TRAINING and ds["license"] == LIC_NO_RETRAIN:
                violations.append({
                    "dataset_id": did,
                    "reason": "许可禁止再训练，不得用于训练运行",
                })
                continue
            # 有条件通过的质检结论进入快照，供审计与复现者知晓
            snapshot[did] = {
                "license": ds["license"],
                "license_expires": ds["license_expires"],
                "grant_id": grant["grant_id"],
                "qc_conclusion": qc["conclusion"],
                "qc_evidence_uri": qc.get("evidence_uri", ""),
                "checksum": ds["checksum"],
                "gene_namespace": ds["gene_namespace"],
                "species": ds["species"],
            }
        if violations:
            raise LedgerError("AUTH_DENIED", "许可/质检闸门未通过",
                              {"violations": violations})
        return snapshot

    # ------------------------------------------------------------------ 运行

    def start_run(self, run_id: str, *, project_id: str, run_type: str,
                  dataset_ids: list[str], code_version: str, image_digest: str,
                  params: dict, mapping_id: str | None = None,
                  model_version_id: str | None = None,
                  derived_from: str | None = None,
                  actor: str = "平台") -> dict:
        """启动训练或推理运行（也用于失败后的第一次登记）。

        训练运行产出模型版本（可 ``derived_from`` 旧模型，旧版本保持不变）；
        推理运行必须基于一个未受影响的正式模型版本，产出嵌入版本。
        """
        with self._lock:
            rid = self._require_id(run_id, "run_id")
            if rid in self.runs:
                raise LedgerError("CONFLICT",
                                  f"运行 {rid} 已存在，失败重试请使用 retry_run")
            self._require_existing(self.projects, project_id, "项目")
            self._require_vocab("运行类型", run_type)
            self._require_existing(self.code_versions, code_version, "代码版本")
            if not isinstance(params, dict):
                raise LedgerError("BAD_REQUEST", "params 必须是对象")

            if run_type == RUN_INFERENCE:
                if not model_version_id:
                    raise LedgerError("BAD_REQUEST",
                                      "推理运行必须指定 model_version_id")
                model = self._require_existing(
                    self.versions, model_version_id, "模型版本")
                if model["kind"] != ART_MODEL:
                    raise LedgerError("BAD_REQUEST",
                                      f"{model_version_id} 不是模型版本")
                if model["status"] != ART_FORMAL:
                    raise LedgerError(
                        "AUTH_DENIED",
                        f"模型版本 {model_version_id} 状态为 {model['status']}，"
                        "不得用于新推理",
                        {"version_status": model["status"]})
            else:
                if derived_from is not None:
                    parent = self._require_existing(
                        self.versions, derived_from, "父模型版本")
                    if parent["kind"] != ART_MODEL:
                        raise LedgerError("BAD_REQUEST",
                                          "derived_from 必须指向模型版本")

            at = self._clock()
            snapshot = self._authorize_inputs(
                dataset_ids, project_id, run_type, at)

            species = {self.datasets[d]["species"] for d in dataset_ids}
            if run_type == RUN_INFERENCE:
                species |= set(self.versions[model_version_id]["species_set"])
            if len(species) > 1:
                if not mapping_id:
                    raise LedgerError(
                        "MAPPING_REQUIRED",
                        "跨物种运行必须显式指定基因标识映射 mapping_id",
                        {"species_set": sorted(species)})
                mapping = self._require_existing(self.mappings, mapping_id, "映射")
                if mapping["status"] == ART_AFFECTED:
                    raise LedgerError(
                        "AUTH_DENIED", f"映射 {mapping_id} 已受影响，不得使用")
                if not self._mapping_covers(mapping, species):
                    raise LedgerError(
                        "MAPPING_MISMATCH",
                        f"映射 {mapping_id} 未覆盖全部物种对",
                        {"species_set": sorted(species),
                         "pairs": mapping["species_pairs"]})

            return self._append("RunStarted", {
                "run_id": rid,
                "project_id": project_id,
                "run_type": run_type,
                "inputs": list(dataset_ids),
                "attempt_no": 1,
                "mapping_id": mapping_id,
                "model_version_id": model_version_id,
                "derived_from": derived_from,
                "code_version": code_version,
                "image_digest": image_digest,
                "params": dict(params),
                "params_checksum": hashlib.sha256(
                    _canonical(params)).hexdigest(),
                "species_set": sorted(species),
                "license_snapshot": snapshot,
            }, actor)

    def fail_run(self, run_id: str, reason: str, actor: str = "平台") -> dict:
        """把运行标记为失败可重试。失败不产生任何版本，可安全重试。"""
        with self._lock:
            run = self._require_existing(self.runs, run_id, "运行")
            if run["status"] != ST_RUNNING:
                raise LedgerError(
                    "BAD_STATE",
                    f"运行 {run_id} 当前状态 {run['status']}，无法标记失败",
                    {"status": run["status"]})
            self._require_id(reason, "reason")
            return self._append("RunFailed", {
                "run_id": run_id, "reason": reason}, actor)

    def retry_run(self, run_id: str, actor: str = "平台") -> dict:
        """安全重试：在同一运行身份下追加新一次尝试。

        重试用运行启动时记录的同一批输入、参数、代码版本与许可快照重跑，
        不新建运行、不复制谱系；只有一次成功，因此绝不会出现两个正式版本。
        """
        with self._lock:
            run = self._require_existing(self.runs, run_id, "运行")
            if run["status"] != ST_FAILED:
                raise LedgerError(
                    "BAD_STATE",
                    f"运行 {run_id} 状态为 {run['status']}，只有失败可重试的"
                    "运行允许重试",
                    {"status": run["status"]})
            # 重试是一次新的执行：许可、授权、质检、映射、模型状态全部按
            # 当下重新校验（尝试之间许可可能已撤销或到期、映射可能已失效）。
            at = self._clock()
            snapshot = self._authorize_inputs(
                run["dataset_ids"], run["project_id"], run["run_type"], at)
            species = set(self._run_species(run))
            if run["mapping_id"]:
                mapping = self.mappings[run["mapping_id"]]
                if mapping["status"] == ART_AFFECTED:
                    raise LedgerError(
                        "AUTH_DENIED",
                        f"映射 {run['mapping_id']} 已受影响，重试被阻止")
                if len(species) > 1 and not self._mapping_covers(mapping, species):
                    raise LedgerError("MAPPING_MISMATCH", "映射不再覆盖物种对")
            if run["run_type"] == RUN_INFERENCE:
                model = self.versions[run["model_version_id"]]
                if model["status"] != ART_FORMAL:
                    raise LedgerError(
                        "AUTH_DENIED",
                        f"模型版本 {model['version_id']} 状态为 "
                        f"{model['status']}，重试被阻止")
            attempt_no = len(run["attempts"]) + 1
            return self._append("RunStarted", {
                "run_id": run_id,
                "project_id": run["project_id"],
                "run_type": run["run_type"],
                "inputs": list(run["dataset_ids"]),
                "attempt_no": attempt_no,
                "mapping_id": run["mapping_id"],
                "model_version_id": run["model_version_id"],
                "derived_from": run["derived_from"],
                "code_version": run["code_version"],
                "image_digest": run["image_digest"],
                "params": dict(run["params"]),
                "params_checksum": hashlib.sha256(
                    _canonical(run["params"])).hexdigest(),
                "species_set": self._run_species(run),
                "license_snapshot": snapshot,
            }, actor)

    def _run_species(self, run: dict) -> list[str]:
        species = {self.datasets[d]["species"] for d in run["dataset_ids"]}
        if run["model_version_id"]:
            species |= set(self.versions[run["model_version_id"]]["species_set"])
        return sorted(species)

    def complete_run(self, run_id: str, version_id: str, checksum: str,
                     metrics: dict | None = None, actor: str = "平台") -> dict:
        """完成运行并晋升唯一正式版本（幂等）。

        * 成功晋升前再次检查授权：运行期间授权被撤销则拒绝晋升并标记受影响；
        * 同一 run_id 重复完成直接返回既有正式版本，绝不产生第二个版本；
        * 新数据派生新模型请另开训练运行（``derived_from``），旧版本不动。
        """
        with self._lock:
            run = self._require_existing(self.runs, run_id, "运行")
            existing = self.run_versions.get(run_id)
            if existing:
                # 幂等返回：调用方重试/重复回调不会制造第二版本
                for event in reversed(self._events):
                    if (event["type"] == "VersionPromoted"
                            and event["payload"]["version_id"] == existing):
                        return copy.deepcopy(event)
                raise LedgerError("BAD_STATE", "内部错误：找不到晋升事件")
            if run["status"] == ST_AFFECTED:
                raise LedgerError(
                    "AUTH_DENIED",
                    f"运行 {run_id} 已受授权撤销影响，不得晋升正式版本")
            if run["status"] != ST_RUNNING:
                raise LedgerError(
                    "BAD_STATE",
                    f"运行 {run_id} 当前状态 {run['status']}，无法完成",
                    {"status": run["status"]})

            # 完成时复核授权仍有效（许可快照只证明启动时合规）
            at = self._clock()
            for did in run["dataset_ids"]:
                grant = self._valid_grant(did, run["project_id"], at)
                if grant is None:
                    self._append("RunFlagged", {
                        "run_id": run_id,
                        "dataset_id": did,
                        "reason": "运行期间授权失效，禁止晋升正式版本",
                    }, actor)
                    raise LedgerError(
                        "AUTH_DENIED",
                        f"数据集 {did} 授权已失效，运行标记为已受影响")

            vid = self._require_id(version_id, "version_id")
            if vid in self.versions:
                raise LedgerError("CONFLICT", f"版本 {vid} 已存在")
            self._require_id(checksum, "checksum")
            kind = ART_MODEL if run["run_type"] == RUN_TRAINING else ART_EMBEDDING
            self._append("VersionRegistered", {
                "version_id": vid,
                "kind": kind,
                "run_id": run_id,
                "project_id": run["project_id"],
                "derived_from": run["derived_from"],
                "model_version_id": run["model_version_id"],
                "mapping_id": run["mapping_id"],
                "checksum": checksum,
                "metrics": metrics or {},
                "species_set": self._run_species(run),
                "params_checksum": hashlib.sha256(
                    _canonical(run["params"])).hexdigest(),
            }, actor)
            self._append("RunCompleted", {
                "run_id": run_id, "version_id": vid}, actor)
            self._append("VersionPromoted", {
                "version_id": vid, "run_id": run_id}, actor)
            return self.events()[-1]

    # ------------------------------------------------------------- 查询结论

    def query_comparison(self, query_id: str, *, project_id: str,
                         query_dataset_id: str, embedding_version_id: str,
                         params: dict, mapping_id: str | None = None,
                         actor: str = "研究者") -> dict:
        """发起细胞相似性比较查询。

        跨物种查询（查询数据物种与参照嵌入覆盖物种不一致）必须显式给出
        mapping_id 与嵌入/模型版本，三者逐字落库；映射须覆盖相关物种对且
        未受影响。授权撤销后查询同样被拒绝。
        """
        with self._lock:
            qid = self._require_id(query_id, "query_id")
            if qid in self.queries:
                raise LedgerError("CONFLICT", f"查询 {qid} 已存在")
            self._require_existing(self.projects, project_id, "项目")
            ds = self._require_existing(self.datasets, query_dataset_id, "数据集")
            emb = self._require_existing(
                self.versions, embedding_version_id, "嵌入版本")
            if emb["kind"] != ART_EMBEDDING:
                raise LedgerError("BAD_REQUEST",
                                  f"{embedding_version_id} 不是嵌入版本")
            if emb["status"] != ART_FORMAL:
                raise LedgerError(
                    "AUTH_DENIED",
                    f"嵌入版本 {embedding_version_id} 状态 {emb['status']}，"
                    "不得用于比较查询",
                    {"version_status": emb["status"]})
            if not isinstance(params, dict):
                raise LedgerError("BAD_REQUEST", "params 必须是对象")

            # 查询方对查询数据仍需有效授权；嵌入所依运行的项目授权也复核
            if self._valid_grant(query_dataset_id, project_id,
                                 self._clock()) is None:
                raise LedgerError(
                    "AUTH_DENIED",
                    f"项目 {project_id} 对查询数据 {query_dataset_id} 无有效授权")
            emb_run = self.runs[emb["run_id"]]
            for did in emb_run["dataset_ids"]:
                if self._valid_grant(did, emb_run["project_id"],
                                     self._clock()) is None:
                    raise LedgerError(
                        "AUTH_DENIED",
                        f"参照嵌入的输入 {did} 授权已失效，查询被阻止")

            model_version_id = emb.get("model_version_id")
            model = self.versions.get(model_version_id) if model_version_id else None
            species = set(emb["species_set"]) | {ds["species"]}
            cross = len(species) > 1
            mapping_used = None
            if cross:
                if not mapping_id:
                    raise LedgerError(
                        "MAPPING_REQUIRED",
                        "跨物种比较查询必须显式指定基因标识映射 mapping_id",
                        {"species_set": sorted(species)})
                mapping = self._require_existing(self.mappings, mapping_id, "映射")
                if mapping["status"] == ART_AFFECTED:
                    raise LedgerError(
                        "AUTH_DENIED", f"映射 {mapping_id} 已受影响，不得使用")
                if not self._mapping_covers(mapping, species):
                    raise LedgerError(
                        "MAPPING_MISMATCH",
                        f"映射 {mapping_id} 未覆盖查询所需物种对",
                        {"species_set": sorted(species)})
                # 查询所用映射必须与嵌入构建时的映射逐字一致，
                # 否则「用哪套映射得到的结论」就无法唯一确定
                emb_mapping_id = emb.get("mapping_id")
                if emb_mapping_id and emb_mapping_id != mapping_id:
                    raise LedgerError(
                        "MAPPING_MISMATCH",
                        f"查询映射 {mapping_id} 与嵌入构建时使用的映射 "
                        f"{emb_mapping_id} 不一致",
                        {"query_mapping": mapping_id,
                         "embedding_mapping": emb_mapping_id})
                mapping_used = {
                    "mapping_id": mapping_id,
                    "name": mapping["name"],
                    "source": mapping["source"],
                    "checksum": mapping["checksum"],
                }

            return self._append("ComparisonQueried", {
                "query_id": qid,
                "project_id": project_id,
                "query_dataset_id": query_dataset_id,
                "query_species": ds["species"],
                "embedding_version_id": embedding_version_id,
                "model_version_id": model_version_id,
                "mapping": mapping_used,
                "cross_species": cross,
                "params": dict(params),
                "params_checksum": hashlib.sha256(
                    _canonical(params)).hexdigest(),
                "code_version": emb_run["code_version"],
            }, actor)

    def record_conclusion(self, conclusion_id: str, *, query_id: str,
                          hypothesis: str, cell_pair: dict,
                          evidence_level: str = "模型推断",
                          actor: str = "研究者") -> dict:
        """记录由查询产生的细胞相似性结论（初始证据等级默认为模型推断）。"""
        with self._lock:
            cid = self._require_id(conclusion_id, "conclusion_id")
            if cid in self.conclusions:
                raise LedgerError("CONFLICT", f"结论 {cid} 已存在")
            self._require_existing(self.queries, query_id, "查询")
            self._require_id(hypothesis, "hypothesis")
            self._require_vocab("证据等级", evidence_level)
            return self._append("ConclusionRecorded", {
                "conclusion_id": cid,
                "query_id": query_id,
                "hypothesis": hypothesis,
                "cell_pair": cell_pair,
                "evidence_level": evidence_level,
            }, actor)

    def annotate(self, conclusion_id: str, researcher: str, text: str,
                 actor: str | None = None) -> dict:
        """研究者批注：只追加，不同意见通过新批注表达，不改写旧批注。"""
        with self._lock:
            self._require_existing(self.conclusions, conclusion_id, "结论")
            self._require_id(researcher, "researcher")
            self._require_id(text, "text")
            return self._append("AnnotationAdded", {
                "conclusion_id": conclusion_id,
                "researcher": researcher,
                "text": text,
            }, actor or researcher)

    def add_validation(self, conclusion_id: str, *, level: str,
                       experiment_ref: str, supports: bool, notes: str = "",
                       actor: str = "研究者") -> dict:
        """记录后续验证证据（计算复核 / 实验验证 / 同行确认）。"""
        with self._lock:
            self._require_existing(self.conclusions, conclusion_id, "结论")
            self._require_vocab("证据等级", level)
            self._require_id(experiment_ref, "experiment_ref")
            if not isinstance(supports, bool):
                raise LedgerError("BAD_REQUEST", "supports 必须为布尔值")
            return self._append("ValidationRecorded", {
                "conclusion_id": conclusion_id,
                "level": level,
                "experiment_ref": experiment_ref,
                "supports": supports,
                "notes": notes,
            }, actor)

    # ------------------------------------------------------------- 追溯视图

    def current_evidence_level(self, conclusion_id: str) -> str:
        """结论当前证据等级：初始等级与支持性验证中的最高等级。

        等级只升（按词表顺序）记录在视图里；历史事件中的初始等级不变。
        """
        conclusion = self._require_existing(
            self.conclusions, conclusion_id, "结论")
        levels = self.vocab["证据等级"]
        rank = {name: i for i, name in enumerate(levels)}
        current = conclusion["evidence_level"]
        for v in self.validations.get(conclusion_id, []):
            if v["supports"] and rank[v["level"]] > rank[current]:
                current = v["level"]
        return current

    def _version_trace(self, version_id: str) -> dict:
        version = self.versions[version_id]
        run = self.runs[version["run_id"]]
        trace = {
            "version_id": version_id,
            "kind": version["kind"],
            "status": version["status"],
            "checksum": version["checksum"],
            "registered_at": version["registered_at"],
            "derived_from": version["derived_from"],
            "produced_by_run": {
                "run_id": run["run_id"],
                "run_type": run["run_type"],
                "status": run["status"],
                "project_id": run["project_id"],
                "code_version": run["code_version"],
                "image_digest": run["image_digest"],
                "params": run["params"],
                "params_checksum": next(
                    (e["payload"].get("params_checksum")
                     for e in self._events
                     if e["type"] == "RunStarted"
                     and e["payload"]["run_id"] == run["run_id"]), None),
                "attempts": run["attempts"],
                "mapping_id": run["mapping_id"],
                "license_snapshot": run["license_snapshot"],
            },
            "inputs": [],
        }
        if version.get("flag_reason"):
            trace["flag_reason"] = version["flag_reason"]
        for did in run["dataset_ids"]:
            ds = self.datasets[did]
            entry = {
                "dataset_id": did,
                "species": ds["species"],
                "tissue": ds["tissue"],
                "disease_state": ds["disease_state"],
                "license": ds["license"],
                "gene_namespace": ds["gene_namespace"],
                "sample_naming": ds["sample_naming"],
                "checksum": ds["checksum"],
                "source_uri": ds["source_uri"],
                "qc": self.qc.get(did),
            }
            snap = run["license_snapshot"].get(did)
            if snap:
                entry["authorization_at_run_start"] = snap
            trace["inputs"].append(entry)
        if run["mapping_id"]:
            m = self.mappings[run["mapping_id"]]
            trace["mapping"] = {k: m[k] for k in (
                "mapping_id", "name", "source", "species_pairs",
                "checksum", "status")}
        if version["kind"] == ART_EMBEDDING and version.get("model_version_id"):
            mid = version["model_version_id"]
            trace["based_on_model"] = self._version_trace(mid)
        if version["derived_from"]:
            trace["parent"] = self._version_trace(version["derived_from"])
        return trace

    def trace_conclusion(self, conclusion_id: str) -> dict:
        """从一个细胞相似性结论反向追溯到全部输入与证据。

        返回：结论 → 批注/验证 → 查询（映射+模型+嵌入+参数+代码版本）
        → 嵌入/模型版本 → 运行（尝试史、许可快照）→ 数据集（质检、许可、
        基因命名空间、校验和）及父版本链。
        """
        conclusion = self._require_existing(
            self.conclusions, conclusion_id, "结论")
        query = self.queries[conclusion["query_id"]]
        trace = {
            "conclusion": {
                **conclusion,
                "current_evidence_level": self.current_evidence_level(conclusion_id),
            },
            "annotations": list(self.annotations.get(conclusion_id, [])),
            "validations": list(self.validations.get(conclusion_id, [])),
            "query": {
                k: query[k] for k in (
                    "query_id", "project_id", "queried_at", "cross_species",
                    "query_species", "params", "params_checksum",
                    "code_version", "mapping")},
            "embedding": self._version_trace(query["embedding_version_id"]),
        }
        if query.get("model_version_id"):
            trace["model"] = self._version_trace(query["model_version_id"])
        affected = []
        for node in (trace["embedding"], trace.get("model")):
            if node and node["status"] == ART_AFFECTED:
                affected.append(node["version_id"])
        trace["affected_artifacts"] = affected
        return trace

    # ------------------------------------------------------------- 只读视图

    def get_dataset(self, dataset_id: str) -> dict:
        with self._lock:
            ds = copy.deepcopy(self._require_existing(
                self.datasets, dataset_id, "数据集"))
            ds["qc"] = copy.deepcopy(self.qc.get(dataset_id))
            ds["grants"] = [
                copy.deepcopy(self.grants[gid])
                for gid in self.dataset_grants.get(dataset_id, [])]
            return ds

    def get_run(self, run_id: str) -> dict:
        with self._lock:
            return copy.deepcopy(self._require_existing(
                self.runs, run_id, "运行"))

    def get_version(self, version_id: str) -> dict:
        with self._lock:
            return copy.deepcopy(self._require_existing(
                self.versions, version_id, "产物版本"))

    def affected_artifacts(self) -> dict:
        """列出撤销级联标记的全部受影响产物（供合规仪表盘）。"""
        return {
            "versions": [{"version_id": v["version_id"], "kind": v["kind"],
                          "reason": v.get("flag_reason", "")}
                         for v in self.versions.values()
                         if v["status"] == ART_AFFECTED],
            "runs": [{"run_id": r["run_id"], "reason": r.get("flag_reason", "")}
                     for r in self.runs.values()
                     if r["status"] == ST_AFFECTED],
            "mappings": [{"mapping_id": m["mapping_id"],
                          "reason": m.get("flag_reason", "")}
                         for m in self.mappings.values()
                         if m["status"] == ART_AFFECTED],
        }

    # ------------------------------------------------------------- 审计导出

    def events(self) -> list[dict]:
        """返回全部审计事件（只读拷贝）。任何命令都不会删除事件。"""
        with self._lock:
            return copy.deepcopy(self._events)

    def snapshot(self) -> dict:
        """导出当前物化状态（调试 / 状态接口用）。"""
        with self._lock:
            return copy.deepcopy({
                "projects": self.projects,
                "datasets": self.datasets,
                "qc": self.qc,
                "grants": self.grants,
                "mappings": self.mappings,
                "code_versions": self.code_versions,
                "runs": self.runs,
                "versions": self.versions,
                "queries": self.queries,
                "conclusions": self.conclusions,
                "annotations": dict(self.annotations),
                "validations": dict(self.validations),
            })


def replay(events: list[dict], vocab: dict | None = None,
           clock=utc_now) -> Ledger:
    """从事件列表重建账本并校验哈希链。

    用于从 JSONL 审计文件恢复：任何事件被篡改或删除导致断链都会抛
    :class:`TamperError`。
    """
    ledger = Ledger(vocab=vocab, clock=clock)
    prev = ""
    for event in events:
        expected = event_hash(
            prev, event["ts"], event["actor"], event["type"], event["payload"])
        if event["prev_hash"] != prev or event["hash"] != expected:
            raise TamperError(
                "TAMPERED",
                f"事件 #{event.get('seq')} 哈希链校验失败：审计记录可能被篡改",
                {"seq": event.get("seq")})
        with ledger._lock:
            ledger._events.append(event)
            ledger._prev_hash = event["hash"]
            ledger._fold(event)
        prev = event["hash"]
    return ledger
