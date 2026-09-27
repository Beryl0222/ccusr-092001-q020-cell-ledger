"""审计事件的 JSONL 持久化。

每行一个事件 JSON。写入采用追加 + 立刻 flush/fsync；读取时保持顺序，
由 :func:`ledger.replay` 校验哈希链。服务启动时重放，崩溃后可无损恢复。
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class JsonlStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        events = []
        with open(self.path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"审计文件 {self.path} 第 {line_no} 行不是合法 JSON"
                    ) from exc
        return events

    def append_many(self, events: list[dict]) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            for event in events:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
