# 跨物种细胞图谱实验账本

项目统一记录单细胞数据许可、计算运行和研究证据，便于不同实验室复现实验谱系。`domain.json` 保存基础状态与证据等级。

运行 `python3 service.py --check` 检查配置，执行 `python3 -m unittest -v` 验证服务身份；服务启动后提供 `/health`。
