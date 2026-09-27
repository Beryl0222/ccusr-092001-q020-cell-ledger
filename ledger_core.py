"""跨物种细胞图谱实验账本的核心领域逻辑。

设计要点
--------
* 事件溯源：所有状态变化先写入只追加的 JSONL 事件日志，再应用到内存状态；
  重启时重放日志恢复状态。审计记录永不删除（许可撤销后仍保留合规审计）。
* 词表（数据许可 / 运行状态 / 证据等级）以 ``domain.json`` 为单一事实来源，
  代码只引用其中的取值，不另造同义词。
* 不变量：
  - 受限数据（非“开放研究”许可）必须持有有效授权才能进入训练运行；
  - 许可撤销后阻止新运行，并把受影响产物标记为“已受影响”，历史记录保留；
  - 失败运行可安全重试，且一次运行至多产出一个正式版本（幂等）；
  - 跨物种运行与查询必须显式声明基因映射与模型；
  - 已产出的正式版本不可改写，新数据只能派生新版本。
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DOMAIN_PATH = Path(__file__).resolve().parent / "domain.json"


# ---------------------------------------------------------------------------
# 词表
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Vocab:
    """domain.json 中的受控词表。"""

    licenses: tuple
    run_statuses: tuple
    evidence_levels: tuple

    @classmethod
    def load(cls, path: Path | str = DOMAIN_PATH) -> "Vocab":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            licenses=tuple(data["数据许可"]),
            run_statuses=tuple(data["运行状态"]),
            evidence_levels=tuple(data["证据等级"]),
        )


VOCAB = Vocab.load()

# 数据许可（取自 domain.json，便于阅读处起别名）
LIC_OPEN = "开放研究"
LIC_PROJECT = "项目内使用"
LIC_NO_RETRAIN = "禁止再训练"
LIC_TIME_LIMITED = "限期使用"
LIC_REVOKED = "已撤销"

# 运行状态（domain.json）
ST_PENDING = "待校验"
ST_QUEUED = "排队"
ST_RUNNING = "运行中"
ST_FAILED = "失败可重试"
ST_OFFICIAL = "正式版本"
ST_AFFECTED = "已受影响"

# 证据等级（domain.json，按可信度升序）
EV_MODEL = "模型推断"
EV_COMPUTED = "计算复核"
EV_EXPERIMENT = "实验验证"
EV_PEER = "同行确认"

# 运行类型与授权范围（命令参数，与 domain.json 词表正交）
RUN_TRAINING = "训练"
RUN_INFERENCE = "推理"
SCOPE_TRAINING = "训练"
SCOPE_INFERENCE = "推理"

# 结论状态（复用运行状态词表中的“已受影响”）
CONCLUSION_VALID = "有效"
CONCLUSION_AFFECTED = ST_AFFECTED


# ---------------------------------------------------------------------------
# 错误
# ---------------------------------------------------------------------------


class LedgerError(Exception):
    """账本业务错误基类。"""


class ValidationError(LedgerError):
    """输入不合法（缺字段、未知词表取值等）。"""


class NotFoundError(LedgerError):
    """引用的实体不存在。"""


class PermissionDeniedError(LedgerError):
    """许可 / 授权不允许该操作。"""


class ConflictError(LedgerError):
    """与当前状态冲突（非法状态迁移、重复正式版本等）。"""


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _require(payload: dict, *fields: str) -> None:
    missing = [f for f in fields if payload.get(f) in (None, "", [])]
    if missing:
        raise ValidationError(f"缺少必填字段: {', '.join(missing)}")


def _parse_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValidationError(f"时间格式非法: {value!r}") from exc


# ---------------------------------------------------------------------------
# 事件存储
# ---------------------------------------------------------------------------


class EventStore:
    """只追加的 JSONL 事件日志；path 为 None 时仅保存在内存。"""

    def __init__(self, path: Path | str | None = None):
        self._path = Path(path) if path else None
        self._events: list[dict] = []
        if self._path and self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    self._events.append(json.loads(line))

    def append(self, event: dict) -> None:
        self._events.append(event)
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def events(self) -> list[dict]:
        return list(self._events)


# ---------------------------------------------------------------------------
# 账本
# ---------------------------------------------------------------------------


class Ledger:
    """实验账本：命令入口 + 事件重放 + 只读查询。"""

    def __init__(self, path: Path | str | None = None, vocab: Vocab = VOCAB):
        self.vocab = vocab
        self._store = EventStore(path)
        self._lock = threading.RLock()
        self._reset_state()
        for event in self._store.events():
            self._apply(event)

    # -- 状态 ---------------------------------------------------------------

    def _reset_state(self) -> None:
        self.datasets: dict[str, dict] = {}
        self.qc_records: list[dict] = []
        self.mappings: dict[str, dict] = {}
        self.models: dict[str, dict] = {}
        self.grants: dict[tuple, dict] = {}
        self.runs: dict[str, dict] = {}
        self.versions: dict[str, dict] = {}
        self.queries: dict[str, dict] = {}
        self.conclusions: dict[str, dict] = {}
        self.annotations: list[dict] = []
        self.validations: list[dict] = []
        self.rejections: list[dict] = []

    def _apply(self, event: dict) -> None:
        handler = getattr(self, f"_on_{event['type']}", None)
        if handler is None:
            raise ValidationError(f"未知事件类型: {event['type']}")
        handler(event)

    def _emit(self, event_type: str, **fields) -> dict:
        event = {"type": event_type, "time": _now(), **fields}
        self._store.append(event)
        self._apply(event)
        return event

    # -- 事件应用器 ----------------------------------------------------------

    def _on_dataset_registered(self, e: dict) -> None:
        self.datasets[e["dataset_id"]] = {
            "dataset_id": e["dataset_id"],
            "project_id": e["project_id"],
            "species": e["species"],
            "tissue": e.get("tissue"),
            "disease_state": e.get("disease_state"),
            "license": e["license"],
            "license_expires_at": e.get("license_expires_at"),
            "registered_at": e["time"],
        }

    def _on_dataset_license_revoked(self, e: dict) -> None:
        ds = self.datasets[e["dataset_id"]]
        ds["license"] = LIC_REVOKED
        ds["revoked_at"] = e["time"]
        ds["revoke_reason"] = e.get("reason")

    def _on_qc_recorded(self, e: dict) -> None:
        self.qc_records.append(
            {
                "dataset_id": e["dataset_id"],
                "metrics": e["metrics"],
                "passed": e["passed"],
                "recorded_at": e["time"],
            }
        )

    def _on_grant_issued(self, e: dict) -> None:
        key = (e["dataset_id"], e["project_id"], e["scope"])
        self.grants[key] = {
            "grant_id": e["grant_id"],
            "dataset_id": e["dataset_id"],
            "project_id": e["project_id"],
            "scope": e["scope"],
            "expires_at": e.get("expires_at"),
            "issued_at": e["time"],
            "revoked": False,
        }

    def _on_grant_revoked(self, e: dict) -> None:
        grant = self.grants.get((e["dataset_id"], e["project_id"], e["scope"]))
        if grant:
            grant["revoked"] = True
            grant["revoked_at"] = e["time"]

    def _on_mapping_registered(self, e: dict) -> None:
        self.mappings[e["mapping_id"]] = {
            "mapping_id": e["mapping_id"],
            "name": e["name"],
            "source_species": e["source_species"],
            "target_species": e["target_species"],
            "version": e["version"],
            "covered_species": list(e["covered_species"]),
            "registered_at": e["time"],
        }

    def _on_model_registered(self, e: dict) -> None:
        self.models[e["model_id"]] = {
            "model_id": e["model_id"],
            "name": e["name"],
            "code_version": e["code_version"],
            "params": e.get("params", {}),
            "registered_at": e["time"],
        }

    def _on_run_submitted(self, e: dict) -> None:
        self.runs[e["run_id"]] = {
            "run_id": e["run_id"],
            "project_id": e["project_id"],
            "run_type": e["run_type"],
            "dataset_ids": list(e["dataset_ids"]),
            "mapping_id": e.get("mapping_id"),
            "model_id": e["model_id"],
            "code_version": e["code_version"],
            "params": e.get("params", {}),
            "input_snapshot": e["input_snapshot"],
            "status": ST_QUEUED,
            "submitted_at": e["time"],
            "version_id": None,
            "derived_from": e.get("derived_from"),
        }

    def _on_run_rejected(self, e: dict) -> None:
        self.rejections.append(
            {
                "project_id": e["project_id"],
                "run_type": e["run_type"],
                "dataset_ids": list(e["dataset_ids"]),
                "reason": e["reason"],
                "rejected_at": e["time"],
            }
        )

    def _on_run_started(self, e: dict) -> None:
        self.runs[e["run_id"]]["status"] = ST_RUNNING

    def _on_run_failed(self, e: dict) -> None:
        run = self.runs[e["run_id"]]
        run["status"] = ST_FAILED
        run["failure_reason"] = e.get("reason")

    def _on_run_retried(self, e: dict) -> None:
        run = self.runs[e["run_id"]]
        run["status"] = ST_QUEUED
        run["failure_reason"] = None

    def _on_run_completed(self, e: dict) -> None:
        self.runs[e["run_id"]]["status"] = ST_OFFICIAL

    def _on_version_promoted(self, e: dict) -> None:
        self.versions[e["version_id"]] = {
            "version_id": e["version_id"],
            "run_id": e["run_id"],
            "project_id": e["project_id"],
            "model_id": e["model_id"],
            "mapping_id": e.get("mapping_id"),
            "dataset_ids": list(e["dataset_ids"]),
            "code_version": e["code_version"],
            "params": e.get("params", {}),
            "derived_from": e.get("derived_from"),
            "status": ST_OFFICIAL,
            "created_at": e["time"],
        }
        self.runs[e["run_id"]]["version_id"] = e["version_id"]

    def _on_artifact_marked_affected(self, e: dict) -> None:
        if e["artifact_kind"] == "version":
            self.versions[e["artifact_id"]]["status"] = ST_AFFECTED
        elif e["artifact_kind"] == "conclusion":
            self.conclusions[e["artifact_id"]]["status"] = CONCLUSION_AFFECTED
        elif e["artifact_kind"] == "run":
            self.runs[e["artifact_id"]]["status"] = ST_AFFECTED

    def _on_query_created(self, e: dict) -> None:
        self.queries[e["query_id"]] = {
            "query_id": e["query_id"],
            "project_id": e["project_id"],
            "version_id": e["version_id"],
            "model_id": e["model_id"],
            "mapping_id": e.get("mapping_id"),
            "cross_species": e["cross_species"],
            "inputs": e["inputs"],
            "created_at": e["time"],
        }

    def _on_conclusion_created(self, e: dict) -> None:
        self.conclusions[e["conclusion_id"]] = {
            "conclusion_id": e["conclusion_id"],
            "project_id": e["project_id"],
            "query_id": e["query_id"],
            "version_id": e["version_id"],
            "model_id": e["model_id"],
            "mapping_id": e.get("mapping_id"),
            "cross_species": e["cross_species"],
            "inputs": e["inputs"],
            "claim": e["claim"],
            "evidence_level": e["evidence_level"],
            "status": CONCLUSION_VALID,
            "created_at": e["time"],
        }

    def _on_annotation_added(self, e: dict) -> None:
        self.annotations.append(
            {
                "annotation_id": e["annotation_id"],
                "target_id": e["target_id"],
                "author": e["author"],
                "text": e["text"],
                "created_at": e["time"],
            }
        )

    def _on_validation_recorded(self, e: dict) -> None:
        self.validations.append(
            {
                "validation_id": e["validation_id"],
                "conclusion_id": e["conclusion_id"],
                "evidence_level": e["evidence_level"],
                "method": e["method"],
                "result": e["result"],
                "recorded_at": e["time"],
            }
        )
        concl = self.conclusions[e["conclusion_id"]]
        if self._evidence_rank(e["evidence_level"]) > self._evidence_rank(
            concl["evidence_level"]
        ):
            concl["evidence_level"] = e["evidence_level"]

    # -- 内部校验 ------------------------------------------------------------

    def _evidence_rank(self, level: str) -> int:
        if level not in self.vocab.evidence_levels:
            raise ValidationError(f"未知证据等级: {level}")
        return self.vocab.evidence_levels.index(level)

    def _get(self, table: dict, key: str, kind: str) -> dict:
        item = table.get(key)
        if item is None:
            raise NotFoundError(f"{kind}不存在: {key}")
        return item

    def _dataset(self, dataset_id: str) -> dict:
        return self._get(self.datasets, dataset_id, "数据集")

    def _run(self, run_id: str) -> dict:
        return self._get(self.runs, run_id, "运行")

    def _version(self, version_id: str) -> dict:
        return self._get(self.versions, version_id, "嵌入版本")

    def _model(self, model_id: str) -> dict:
        return self._get(self.models, model_id, "模型")

    def _mapping(self, mapping_id: str) -> dict:
        return self._get(self.mappings, mapping_id, "基因映射")

    def _conclusion(self, conclusion_id: str) -> dict:
        return self._get(self.conclusions, conclusion_id, "结论")

    def _latest_qc(self, dataset_id: str) -> dict | None:
        records = [q for q in self.qc_records if q["dataset_id"] == dataset_id]
        return records[-1] if records else None

    def _active_grant(self, dataset_id: str, project_id: str, scope: str) -> dict | None:
        grant = self.grants.get((dataset_id, project_id, scope))
        if not grant or grant["revoked"]:
            return None
        if grant.get("expires_at") and _parse_time(grant["expires_at"]) < datetime.now(
            timezone.utc
        ):
            return None
        return grant

    def _check_dataset_usable(
        self, dataset: dict, project_id: str, run_type: str
    ) -> None:
        """许可与授权校验：受限数据不得被未授权项目使用。"""
        ds_id = dataset["dataset_id"]
        lic = dataset["license"]
        if lic == LIC_REVOKED:
            raise PermissionDeniedError(f"数据集 {ds_id} 许可已撤销")
        if lic == LIC_OPEN:
            return
        if lic == LIC_NO_RETRAIN and run_type == RUN_TRAINING:
            raise PermissionDeniedError(f"数据集 {ds_id} 许可为“禁止再训练”")
        if lic == LIC_TIME_LIMITED:
            expires = dataset.get("license_expires_at")
            if not expires:
                raise ValidationError(f"数据集 {ds_id} 缺少许可到期时间")
            if _parse_time(expires) < datetime.now(timezone.utc):
                raise PermissionDeniedError(f"数据集 {ds_id} 许可已过期")
        if lic in (LIC_PROJECT, LIC_TIME_LIMITED, LIC_NO_RETRAIN):
            if self._active_grant(ds_id, project_id, run_type) is None:
                raise PermissionDeniedError(
                    f"数据集 {ds_id} 许可为“{lic}”，项目 {project_id} 缺少有效{run_type}授权"
                )

    def _authorize_inputs(
        self, project_id: str, run_type: str, dataset_ids: list[str], mapping_id
    ) -> None:
        datasets = [self._dataset(d) for d in dataset_ids]
        for ds in datasets:
            self._check_dataset_usable(ds, project_id, run_type)
        species = {ds["species"] for ds in datasets}
        if len(species) > 1:
            if not mapping_id:
                raise ValidationError("跨物种运行必须显式指定基因映射")
            mapping = self._mapping(mapping_id)
            uncovered = species - set(mapping["covered_species"])
            if uncovered:
                raise ValidationError(
                    f"基因映射 {mapping_id} 未覆盖物种: {sorted(uncovered)}"
                )

    def _check_qc(self, dataset_ids: list[str]) -> None:
        for ds_id in dataset_ids:
            qc = self._latest_qc(ds_id)
            if qc is None:
                raise ValidationError(f"数据集 {ds_id} 尚未记录质量检查")
            if not qc["passed"]:
                raise ValidationError(f"数据集 {ds_id} 最近一次质量检查未通过")

    def _reject(self, project_id, run_type, dataset_ids, reason) -> None:
        """被拒绝的运行也写入审计日志。"""
        self._emit(
            "run_rejected",
            project_id=project_id,
            run_type=run_type,
            dataset_ids=list(dataset_ids),
            reason=reason,
        )

    # -- 命令：数据集 / 许可 --------------------------------------------------

    def register_dataset(
        self,
        project_id: str,
        species: str,
        license: str,
        tissue: str | None = None,
        disease_state: str | None = None,
        license_expires_at: str | None = None,
        dataset_id: str | None = None,
    ) -> dict:
        with self._lock:
            if license not in self.vocab.licenses:
                raise ValidationError(f"未知数据许可: {license}")
            if license == LIC_REVOKED:
                raise ValidationError("数据集不能以“已撤销”许可注册")
            if license == LIC_TIME_LIMITED and not license_expires_at:
                raise ValidationError("“限期使用”许可必须给出到期时间")
            if license_expires_at:
                _parse_time(license_expires_at)
            dataset_id = dataset_id or _new_id("ds")
            if dataset_id in self.datasets:
                raise ConflictError(f"数据集已存在: {dataset_id}")
            self._emit(
                "dataset_registered",
                dataset_id=dataset_id,
                project_id=project_id,
                species=species,
                tissue=tissue,
                disease_state=disease_state,
                license=license,
                license_expires_at=license_expires_at,
            )
            return deepcopy(self.datasets[dataset_id])

    def record_qc(self, dataset_id: str, metrics: dict, passed: bool) -> dict:
        with self._lock:
            self._dataset(dataset_id)
            if not isinstance(metrics, dict):
                raise ValidationError("质量检查指标必须是对象")
            self._emit(
                "qc_recorded",
                dataset_id=dataset_id,
                metrics=metrics,
                passed=bool(passed),
            )
            return deepcopy(self.qc_records[-1])

    def issue_grant(
        self,
        dataset_id: str,
        project_id: str,
        scope: str,
        expires_at: str | None = None,
    ) -> dict:
        with self._lock:
            self._dataset(dataset_id)
            if scope not in (SCOPE_TRAINING, SCOPE_INFERENCE):
                raise ValidationError(f"未知授权范围: {scope}")
            if expires_at:
                _parse_time(expires_at)
            grant_id = _new_id("grant")
            self._emit(
                "grant_issued",
                grant_id=grant_id,
                dataset_id=dataset_id,
                project_id=project_id,
                scope=scope,
                expires_at=expires_at,
            )
            return deepcopy(self.grants[(dataset_id, project_id, scope)])

    def revoke_grant(self, dataset_id: str, project_id: str, scope: str) -> dict:
        with self._lock:
            grant = self.grants.get((dataset_id, project_id, scope))
            if grant is None:
                raise NotFoundError("授权不存在")
            if grant["revoked"]:
                return deepcopy(grant)  # 幂等
            self._emit(
                "grant_revoked",
                dataset_id=dataset_id,
                project_id=project_id,
                scope=scope,
            )
            return deepcopy(self.grants[(dataset_id, project_id, scope)])

    def revoke_dataset_license(self, dataset_id: str, reason: str | None = None) -> dict:
        """撤销许可：阻止新运行，级联标记受影响产物，保留全部审计记录。"""
        with self._lock:
            dataset = self._dataset(dataset_id)
            if dataset["license"] == LIC_REVOKED:
                return self.affected_report(dataset_id)  # 幂等
            self._emit("dataset_license_revoked", dataset_id=dataset_id, reason=reason)

            affected_versions: list[str] = []
            affected_conclusions: list[str] = []
            affected_runs: list[str] = []

            # 未完成的运行直接标记为受影响，不再推进。
            for run in self.runs.values():
                if dataset_id in run["dataset_ids"] and run["status"] in (
                    ST_QUEUED,
                    ST_RUNNING,
                    ST_FAILED,
                ):
                    self._emit(
                        "artifact_marked_affected",
                        artifact_kind="run",
                        artifact_id=run["run_id"],
                        cause=f"数据集 {dataset_id} 许可撤销",
                    )
                    affected_runs.append(run["run_id"])

            # 使用该数据集训练的正式版本标记为受影响。
            for version in self.versions.values():
                if (
                    dataset_id in version["dataset_ids"]
                    and version["status"] == ST_OFFICIAL
                ):
                    self._emit(
                        "artifact_marked_affected",
                        artifact_kind="version",
                        artifact_id=version["version_id"],
                        cause=f"数据集 {dataset_id} 许可撤销",
                    )
                    affected_versions.append(version["version_id"])

            # 依赖受影响版本的结论标记为受影响（证据保留，状态降级）。
            for concl in self.conclusions.values():
                if (
                    concl["version_id"] in affected_versions
                    and concl["status"] == CONCLUSION_VALID
                ):
                    self._emit(
                        "artifact_marked_affected",
                        artifact_kind="conclusion",
                        artifact_id=concl["conclusion_id"],
                        cause=f"嵌入版本 {concl['version_id']} 受影响",
                    )
                    affected_conclusions.append(concl["conclusion_id"])

            return self.affected_report(dataset_id)

    # -- 命令：映射 / 模型 ----------------------------------------------------

    def register_mapping(
        self,
        name: str,
        source_species: str,
        target_species: str,
        version: str,
        covered_species: list[str],
        mapping_id: str | None = None,
    ) -> dict:
        with self._lock:
            _require(
                {"name": name, "version": version, "covered_species": covered_species},
                "name",
                "version",
                "covered_species",
            )
            mapping_id = mapping_id or _new_id("map")
            if mapping_id in self.mappings:
                raise ConflictError(f"基因映射已存在: {mapping_id}")
            self._emit(
                "mapping_registered",
                mapping_id=mapping_id,
                name=name,
                source_species=source_species,
                target_species=target_species,
                version=version,
                covered_species=list(covered_species),
            )
            return deepcopy(self.mappings[mapping_id])

    def register_model(
        self,
        name: str,
        code_version: str,
        params: dict | None = None,
        model_id: str | None = None,
    ) -> dict:
        with self._lock:
            _require({"name": name, "code_version": code_version}, "name", "code_version")
            model_id = model_id or _new_id("model")
            if model_id in self.models:
                raise ConflictError(f"模型已存在: {model_id}")
            self._emit(
                "model_registered",
                model_id=model_id,
                name=name,
                code_version=code_version,
                params=params or {},
            )
            return deepcopy(self.models[model_id])

    # -- 命令：运行生命周期 ----------------------------------------------------

    def submit_run(
        self,
        project_id: str,
        run_type: str,
        dataset_ids: list[str],
        model_id: str,
        code_version: str,
        params: dict | None = None,
        mapping_id: str | None = None,
        derived_from: str | None = None,
        run_id: str | None = None,
    ) -> dict:
        """提交训练 / 推理运行。许可与输入校验失败会留下审计记录。"""
        with self._lock:
            if run_type not in (RUN_TRAINING, RUN_INFERENCE):
                raise ValidationError(f"未知运行类型: {run_type}")
            _require(
                {"project_id": project_id, "dataset_ids": dataset_ids,
                 "model_id": model_id, "code_version": code_version},
                "project_id",
                "dataset_ids",
                "model_id",
                "code_version",
            )
            try:
                self._model(model_id)
                if mapping_id:
                    self._mapping(mapping_id)
                if derived_from:
                    self._version(derived_from)
                self._authorize_inputs(project_id, run_type, dataset_ids, mapping_id)
                self._check_qc(dataset_ids)
            except LedgerError as exc:
                self._reject(project_id, run_type, dataset_ids, str(exc))
                raise

            run_id = run_id or _new_id("run")
            if run_id in self.runs:
                raise ConflictError(f"运行已存在: {run_id}")
            snapshot = [
                {
                    "dataset_id": ds["dataset_id"],
                    "species": ds["species"],
                    "license": ds["license"],
                }
                for ds in (self._dataset(d) for d in dataset_ids)
            ]
            self._emit(
                "run_submitted",
                run_id=run_id,
                project_id=project_id,
                run_type=run_type,
                dataset_ids=list(dataset_ids),
                mapping_id=mapping_id,
                model_id=model_id,
                code_version=code_version,
                params=params or {},
                input_snapshot=snapshot,
                derived_from=derived_from,
            )
            return deepcopy(self.runs[run_id])

    def start_run(self, run_id: str) -> dict:
        with self._lock:
            run = self._run(run_id)
            if run["status"] != ST_QUEUED:
                raise ConflictError(f"运行 {run_id} 当前状态为“{run['status']}”，不能启动")
            try:
                self._authorize_inputs(
                    run["project_id"], run["run_type"], run["dataset_ids"], run["mapping_id"]
                )
            except LedgerError as exc:
                self._reject(run["project_id"], run["run_type"], run["dataset_ids"], str(exc))
                raise
            self._emit("run_started", run_id=run_id)
            return deepcopy(self.runs[run_id])

    def fail_run(self, run_id: str, reason: str | None = None) -> dict:
        with self._lock:
            run = self._run(run_id)
            if run["status"] != ST_RUNNING:
                raise ConflictError(f"运行 {run_id} 当前状态为“{run['status']}”，不能标记失败")
            self._emit("run_failed", run_id=run_id, reason=reason)
            return deepcopy(self.runs[run_id])

    def retry_run(self, run_id: str) -> dict:
        """失败运行安全重试：回到排队状态，沿用原运行身份，不产生第二个正式版本。"""
        with self._lock:
            run = self._run(run_id)
            if run["status"] == ST_FAILED:
                self._authorize_inputs(
                    run["project_id"], run["run_type"], run["dataset_ids"], run["mapping_id"]
                )
                self._emit("run_retried", run_id=run_id)
            elif run["status"] in (ST_QUEUED, ST_RUNNING):
                pass  # 幂等：重复重试请求直接返回当前状态
            else:
                raise ConflictError(f"运行 {run_id} 当前状态为“{run['status']}”，不能重试")
            return deepcopy(self.runs[run_id])

    def complete_run(self, run_id: str) -> dict:
        """完成运行并产出唯一正式版本（幂等：重复完成返回同一版本）。"""
        with self._lock:
            run = self._run(run_id)
            if run["status"] == ST_OFFICIAL and run["version_id"]:
                return deepcopy(self.versions[run["version_id"]])
            if run["status"] != ST_RUNNING:
                raise ConflictError(f"运行 {run_id} 当前状态为“{run['status']}”，不能完成")
            # 完成前复核许可：运行期间许可可能已撤销。
            self._authorize_inputs(
                run["project_id"], run["run_type"], run["dataset_ids"], run["mapping_id"]
            )
            version_id = _new_id("ver")
            self._emit("run_completed", run_id=run_id)
            self._emit(
                "version_promoted",
                version_id=version_id,
                run_id=run_id,
                project_id=run["project_id"],
                model_id=run["model_id"],
                mapping_id=run["mapping_id"],
                dataset_ids=list(run["dataset_ids"]),
                code_version=run["code_version"],
                params=run["params"],
                derived_from=run["derived_from"],
            )
            return deepcopy(self.versions[version_id])

    def derive_version(
        self,
        base_version_id: str,
        project_id: str,
        run_type: str,
        dataset_ids: list[str],
        code_version: str,
        params: dict | None = None,
        mapping_id: str | None = None,
    ) -> dict:
        """在既有版本基础上派生新版本（新数据到来时），不改写旧版本。"""
        with self._lock:
            base = self._version(base_version_id)
            mapping_id = mapping_id if mapping_id is not None else base["mapping_id"]
            run = self.submit_run(
                project_id=project_id,
                run_type=run_type,
                dataset_ids=dataset_ids,
                model_id=base["model_id"],
                code_version=code_version,
                params=params,
                mapping_id=mapping_id,
                derived_from=base_version_id,
            )
            return run

    # -- 命令：查询 / 结论 / 批注 / 验证 ---------------------------------------

    def create_query(
        self,
        project_id: str,
        version_id: str,
        inputs: list[dict],
        model_id: str | None = None,
        mapping_id: str | None = None,
        query_id: str | None = None,
    ) -> dict:
        """比较查询。跨物种查询必须显式给出所用映射与模型。"""
        with self._lock:
            _require({"project_id": project_id, "inputs": inputs}, "project_id", "inputs")
            version = self._version(version_id)
            if version["status"] != ST_OFFICIAL:
                raise PermissionDeniedError(
                    f"嵌入版本 {version_id} 状态为“{version['status']}”，不可用于查询"
                )
            model_id = model_id or version["model_id"]
            if model_id != version["model_id"]:
                raise ValidationError(
                    f"查询模型 {model_id} 与版本训练模型 {version['model_id']} 不一致"
                )

            species: set[str] = set()
            for item in inputs:
                if not isinstance(item, dict) or "species" not in item:
                    raise ValidationError("查询输入必须包含 species 字段")
                species.add(item["species"])
            cross_species = len(species) > 1
            if cross_species:
                # 跨物种查询必须显式声明所用映射，且与版本训练映射一致。
                if not mapping_id:
                    raise ValidationError("跨物种查询必须显式指定基因映射")
                if version["mapping_id"] and mapping_id != version["mapping_id"]:
                    raise ValidationError(
                        f"查询映射 {mapping_id} 与版本训练映射 "
                        f"{version['mapping_id']} 不一致"
                    )
                mapping = self._mapping(mapping_id)
                uncovered = species - set(mapping["covered_species"])
                if uncovered:
                    raise ValidationError(
                        f"基因映射 {mapping_id} 未覆盖物种: {sorted(uncovered)}"
                    )
            elif mapping_id is None:
                mapping_id = version["mapping_id"]

            query_id = query_id or _new_id("qry")
            self._emit(
                "query_created",
                query_id=query_id,
                project_id=project_id,
                version_id=version_id,
                model_id=model_id,
                mapping_id=mapping_id,
                cross_species=cross_species,
                inputs=deepcopy(inputs),
            )
            return deepcopy(self.queries[query_id])

    def create_conclusion(
        self,
        project_id: str,
        query_id: str,
        claim: str,
        inputs: list[dict],
        evidence_level: str = EV_MODEL,
        conclusion_id: str | None = None,
    ) -> dict:
        """记录细胞相似性结论（默认证据等级为“模型推断”）。"""
        with self._lock:
            _require({"claim": claim, "inputs": inputs}, "claim", "inputs")
            self._evidence_rank(evidence_level)
            query = self._get(self.queries, query_id, "查询")
            conclusion_id = conclusion_id or _new_id("concl")
            self._emit(
                "conclusion_created",
                conclusion_id=conclusion_id,
                project_id=project_id,
                query_id=query_id,
                version_id=query["version_id"],
                model_id=query["model_id"],
                mapping_id=query["mapping_id"],
                cross_species=query["cross_species"],
                inputs=deepcopy(inputs),
                claim=claim,
                evidence_level=evidence_level,
            )
            return deepcopy(self.conclusions[conclusion_id])

    def add_annotation(self, target_id: str, author: str, text: str) -> dict:
        """研究者批注，可附加到任意实体。"""
        with self._lock:
            _require({"author": author, "text": text}, "author", "text")
            known = (
                self.datasets,
                self.runs,
                self.versions,
                self.queries,
                self.conclusions,
                self.mappings,
                self.models,
            )
            if not any(target_id in table for table in known):
                raise NotFoundError(f"批注目标不存在: {target_id}")
            annotation_id = _new_id("note")
            self._emit(
                "annotation_added",
                annotation_id=annotation_id,
                target_id=target_id,
                author=author,
                text=text,
            )
            return deepcopy(self.annotations[-1])

    def record_validation(
        self,
        conclusion_id: str,
        evidence_level: str,
        method: str,
        result: str,
    ) -> dict:
        """记录后续验证证据；结论证据等级只升不降。"""
        with self._lock:
            _require({"method": method, "result": result}, "method", "result")
            concl = self._conclusion(conclusion_id)
            self._evidence_rank(evidence_level)
            validation_id = _new_id("val")
            self._emit(
                "validation_recorded",
                validation_id=validation_id,
                conclusion_id=conclusion_id,
                evidence_level=evidence_level,
                method=method,
                result=result,
            )
            return {
                "validation": deepcopy(self.validations[-1]),
                "conclusion": deepcopy(self.conclusions[conclusion_id]),
            }

    # -- 只读查询 --------------------------------------------------------------

    def get_dataset(self, dataset_id: str) -> dict:
        return deepcopy(self._dataset(dataset_id))

    def get_run(self, run_id: str) -> dict:
        return deepcopy(self._run(run_id))

    def get_version(self, version_id: str) -> dict:
        return deepcopy(self._version(version_id))

    def get_conclusion(self, conclusion_id: str) -> dict:
        return deepcopy(self._conclusion(conclusion_id))

    def list_datasets(self) -> list[dict]:
        return deepcopy(list(self.datasets.values()))

    def list_runs(self) -> list[dict]:
        return deepcopy(list(self.runs.values()))

    def list_versions(self) -> list[dict]:
        return deepcopy(list(self.versions.values()))

    def list_annotations(self, target_id: str) -> list[dict]:
        return deepcopy([a for a in self.annotations if a["target_id"] == target_id])

    def affected_report(self, dataset_id: str) -> dict:
        """许可撤销影响面报告（合规审计依据）。"""
        with self._lock:
            self._dataset(dataset_id)
            versions = [
                v["version_id"]
                for v in self.versions.values()
                if dataset_id in v["dataset_ids"] and v["status"] == ST_AFFECTED
            ]
            conclusions = [
                c["conclusion_id"]
                for c in self.conclusions.values()
                if c["version_id"] in versions and c["status"] == CONCLUSION_AFFECTED
            ]
            runs = [
                r["run_id"]
                for r in self.runs.values()
                if dataset_id in r["dataset_ids"] and r["status"] == ST_AFFECTED
            ]
            return {
                "dataset_id": dataset_id,
                "affected_versions": versions,
                "affected_conclusions": conclusions,
                "affected_runs": runs,
            }

    def trace_conclusion(self, conclusion_id: str) -> dict:
        """从结论反向追溯：输入、参数、代码版本、模型、映射与验证证据。"""
        with self._lock:
            concl = self._conclusion(conclusion_id)
            query = self.queries.get(concl["query_id"])
            version = self.versions.get(concl["version_id"])
            run = self.runs.get(version["run_id"]) if version else None
            model = self.models.get(concl["model_id"])
            mapping = (
                self.mappings.get(concl["mapping_id"]) if concl["mapping_id"] else None
            )
            datasets = [
                deepcopy(self.datasets[d])
                for d in (run["dataset_ids"] if run else [])
                if d in self.datasets
            ]
            lineage: list[dict] = []
            cursor = version
            while cursor and cursor.get("derived_from"):
                parent = self.versions.get(cursor["derived_from"])
                if parent is None:
                    break
                lineage.append(deepcopy(parent))
                cursor = parent
            return {
                "conclusion": deepcopy(concl),
                "query": deepcopy(query),
                "version": deepcopy(version),
                "run": deepcopy(run),
                "model": deepcopy(model),
                "mapping": deepcopy(mapping),
                "datasets": datasets,
                "validations": deepcopy(
                    [v for v in self.validations if v["conclusion_id"] == conclusion_id]
                ),
                "annotations": self.list_annotations(conclusion_id),
                "derived_lineage": lineage,
            }
