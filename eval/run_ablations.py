# -*- coding: utf-8 -*-
"""一键复现三组消融实验（默认跑 214 题评测集）。

消融口径（答案关键词命中 kw@5 / Top-5 文档召回 doc@5）：
  baseline   旧切块 + TF-IDF + 无改写   —— 基线
  full       词条切块 + 查询改写 + 密集/词法 RRF 融合   —— 全链路
  -RRF       仅 bge-m3 密集（去掉 RRF 融合）
  -rewrite   去掉查询改写
  -chunking  退回旧切块（去掉词条级切块）

用法（bge-m3 模型路径用环境变量指定）：
  set BGE_M3_MODEL=E:\\models\\bge-m3
  python eval/run_ablations.py --set eval/eval_set_200.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIGS: list[tuple[str, list[str]]] = [
    ("baseline", ["--backend", "tfidf", "--chunking", "naive", "--rewrite", "none"]),
    ("full", ["--backend", "hybrid", "--chunking", "structured", "--rewrite", "basic"]),
    ("-RRF", ["--backend", "bge-m3", "--chunking", "structured", "--rewrite", "basic"]),
    ("-rewrite", ["--backend", "hybrid", "--chunking", "structured", "--rewrite", "none"]),
    ("-chunking", ["--backend", "hybrid", "--chunking", "naive", "--rewrite", "basic"]),
]


def _tag(cfg: list[str]) -> str:
    backend = cfg[cfg.index("--backend") + 1]
    chunking = cfg[cfg.index("--chunking") + 1]
    rewrite = cfg[cfg.index("--rewrite") + 1]
    tag = backend
    if chunking != "naive":
        tag += f"_{chunking}"
    if rewrite != "none":
        tag += "_rw"
    return tag


def main() -> int:
    parser = argparse.ArgumentParser(description="一键复现三组消融实验")
    parser.add_argument("--set", default="eval/eval_set_200.json")
    parser.add_argument("--out", default="eval/output_ablation")
    args = parser.parse_args()

    rows: dict[str, dict] = {}
    for name, cfg in CONFIGS:
        print(f"\n========== {name}: {' '.join(cfg)} ==========", flush=True)
        cmd = [
            sys.executable,
            str(ROOT / "eval" / "local_retrieval_eval.py"),
            *cfg,
            "--set",
            args.set,
            "--out",
            args.out,
        ]
        subprocess.run(cmd, cwd=ROOT, check=True)
        summary_path = ROOT / args.out / f"local_summary_{_tag(cfg)}.json"
        rows[name] = json.loads(summary_path.read_text(encoding="utf-8"))

    full = rows["full"]
    print("\n消融结果（评测集 n=%d）" % full["n"])
    print(f"{'配置':<12}{'kw@5':>10}{'doc@5':>10}")
    for name in [c[0] for c in CONFIGS]:
        print(f"{name:<12}{rows[name]['kw@5'] * 100:>9.2f}%{rows[name]['doc@5'] * 100:>9.2f}%")
    print("\n相对全链路的 kw@5 单项增益：")
    for name in ("-RRF", "-rewrite", "-chunking"):
        delta = (full["kw@5"] - rows[name]["kw@5"]) * 100
        print(f"  {name:<12} +{delta:.2f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
