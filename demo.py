"""端到端演示：跨物种联合研究的一次完整谱系。

场景：两个实验室分别贡献人类与小鼠单细胞数据，在共同坐标空间训练嵌入，
提出跨物种细胞相似性结论并记录验证证据；随后人类数据许可被撤销，
展示合规级联与审计保留。

运行：python3 demo.py
"""

from ledger_core import (
    EV_COMPUTED,
    LIC_OPEN,
    LIC_PROJECT,
    RUN_TRAINING,
    SCOPE_TRAINING,
    Ledger,
)


def main() -> None:
    lg = Ledger(None)  # 演示用内存账本；服务模式下持久化到 ledger.jsonl

    # 1. 数据集接入与质量检查
    ds_human = lg.register_dataset(
        "proj-atlas", "人类", LIC_PROJECT, tissue="血液", disease_state="健康"
    )["dataset_id"]
    ds_mouse = lg.register_dataset(
        "proj-atlas", "小鼠", LIC_OPEN, tissue="脾脏"
    )["dataset_id"]
    for ds in (ds_human, ds_mouse):
        lg.record_qc(ds, {"cells": 12000, "doublet_rate": 0.03}, passed=True)

    # 2. 受限数据授权：人类数据仅授权 proj-atlas 训练
    lg.issue_grant(ds_human, "proj-atlas", SCOPE_TRAINING)

    # 3. 基因映射与模型登记
    mapping = lg.register_mapping(
        "人鼠同源基因映射", "人类", "小鼠", "homologene-2026.1",
        covered_species=["人类", "小鼠"],
    )["mapping_id"]
    model = lg.register_model(
        "跨物种嵌入模型", code_version="git:abc123", params={"dim": 64}
    )["model_id"]

    # 4. 跨物种训练运行 → 唯一正式版本
    run = lg.submit_run(
        "proj-atlas", RUN_TRAINING, [ds_human, ds_mouse], model,
        code_version="git:abc123", mapping_id=mapping,
    )
    lg.start_run(run["run_id"])
    version = lg.complete_run(run["run_id"])
    print("正式版本:", version["version_id"])

    # 5. 跨物种比较查询（显式声明映射与模型）→ 相似性结论
    query = lg.create_query(
        "proj-atlas", version["version_id"],
        inputs=[{"species": "人类", "cell": "CD4-T-001"},
                {"species": "小鼠", "cell": "Cd4-T-117"}],
        mapping_id=mapping,
    )
    conclusion = lg.create_conclusion(
        "proj-atlas", query["query_id"],
        claim="人类 CD4-T-001 与小鼠 Cd4-T-117 转录状态相似",
        inputs=[{"cell": "CD4-T-001"}, {"cell": "Cd4-T-117"}],
    )
    lg.add_annotation(conclusion["conclusion_id"], "王研究员", "建议做跨物种 marker 复核")
    lg.record_validation(
        conclusion["conclusion_id"], EV_COMPUTED, "独立管线重嵌入", "相似性复现"
    )

    # 6. 从结论反向追溯
    trace = lg.trace_conclusion(conclusion["conclusion_id"])
    print("结论证据等级:", trace["conclusion"]["evidence_level"])
    print("训练代码版本:", trace["version"]["code_version"])
    print("输入数据集:", [d["dataset_id"] for d in trace["datasets"]])

    # 7. 人类数据许可被撤销：阻止新运行、级联标记、保留审计
    report = lg.revoke_dataset_license(ds_human, reason="供体撤回同意")
    print("撤销影响面:", report)
    try:
        lg.submit_run("proj-atlas", RUN_TRAINING, [ds_human], model, "git:abc123")
    except Exception as exc:
        print("撤销后新运行被阻止:", exc)
    print("被拒绝运行审计条数:", len(lg.rejections))


if __name__ == "__main__":
    main()
