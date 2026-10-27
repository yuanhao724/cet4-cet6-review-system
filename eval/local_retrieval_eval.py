"""不依赖 RAGFlow 的本地离线检索评测。

三种后端：
1) tfidf   —— 字符 n-gram TF-IDF + 余弦相似度，秒级出结果，作为词法基线。
2) bge-m3  —— 本地 transformers 加载 BAAI/bge-m3，真实语义检索（需下载约 2 GB 模型）。
3) hybrid  —— bge-m3 密集 + TF-IDF 词法双路召回，RRF 融合（默认，修复短 chunk 抢答问题）。

评测口径与 eval_lib 一致：doc@1/3/5（来源文档命中）与 kw@1/3/5（答案关键词命中）。
结果明确标注为“本地离线复现”，不是 RAGFlow 线上指标。

修复项（对应“错题复盘 → 拆 chunking / query 改写”的迭代）：
- structured 切块：识别单词文档里的 HTML 表格，按“每个词条一条”切块（旧实现把整张表当一坨硬切）；
- basic 查询改写：vocab 查询提取英文词头；translation 查询用知识库自带的中英对照做双语扩展。

用法：
    python eval/local_retrieval_eval.py --backend tfidf
    python eval/local_retrieval_eval.py --backend bge-m3
    python eval/local_retrieval_eval.py --backend hybrid            # 默认
    python eval/local_retrieval_eval.py --backend bge-m3 --chunking naive --rewrite none   # 复现旧基线
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import numpy as np

from eval_lib import evaluate_item

KS = (1, 3, 5)
_HTML_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S | re.I)
_HTML_CELL = re.compile(r"<(?:td|th)\b[^>]*>(.*?)</(?:td|th)>", re.S | re.I)
_HTML_TAG = re.compile(r"<[^>]+>")
_WORD_LIKE = re.compile(r"^[A-Za-z][A-Za-z\-'. ]{0,39}$")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_EN_WORD_IN_QUERY = re.compile(r"单词\s*([A-Za-z][A-Za-z\-']*)")
_GLOSSARY_DOC_FRAGS = ("对照表", "高频扣分", "高低级词汇替换")


def _clean_html_cell(html: str) -> str:
    html = (
        html.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
    )
    return _HTML_TAG.sub(" ", html).strip()


def split_table_entries(text: str) -> list[str]:
    """把词汇表（HTML 表格）拆成“word 释义”的逐词条 chunk。"""
    entries: list[str] = []
    for row in _HTML_ROW.finditer(text):
        cells = [c for c in (_clean_html_cell(c) for c in _HTML_CELL.findall(row.group(1))) if c]
        if len(cells) < 2:
            continue
        if any(c in ("序号", "单词", "注音", "释义") for c in cells):
            continue  # 表头行
        word = next((c for c in cells if _WORD_LIKE.match(c)), None)
        cjk_cells = [c for c in cells if _CJK.search(c)]
        meaning = max(cjk_cells, key=len) if cjk_cells else None
        if word and meaning:
            entries.append(f"{word} {meaning}")
        else:
            body = cells[1:] if (cells and re.fullmatch(r"\d+", cells[0])) else cells
            if body:
                entries.append(" ".join(body))
    return entries


def _normalize_zh(s: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", s)


def build_translation_glossary(doc_dir: Path) -> dict[str, str]:
    """从知识库的翻译文档抽取“中文 → 英文”对照，用于 translation 查询的双语扩展。"""
    glossary: dict[str, str] = {}
    trans_dir = doc_dir / "翻译"
    for path in sorted(trans_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = [ln.strip() for ln in text.splitlines()]
        i = 0
        while i < len(lines):
            line = lines[i]
            has_cjk = bool(_CJK.search(line))
            has_latin = bool(re.search(r"[A-Za-z]{2,}", line))
            if has_cjk and has_latin:
                # 对照表风格：剪纸 paper-cutting
                m = re.match(r"^\s*[\u2026.]*\s*([\u4e00-\u9fff][\u4e00-\u9fff（）·、]*?)\s+([A-Za-z][A-Za-z\-() ]+)", line)
                if m:
                    key = _normalize_zh(m.group(1))
                    if len(key) >= 2:
                        glossary.setdefault(key, m.group(2).strip())
            elif has_cjk and not has_latin:
                # 万能句式风格：中文行 + 下一行英文
                if not re.match(r"^[一二三四五六七八九十]+、", line) and "、" not in line[:4]:
                    j = i + 1
                    while j < len(lines) and not lines[j]:
                        j += 1
                    if j < len(lines) and re.search(r"[A-Za-z]{2,}", lines[j]):
                        en = re.sub(r"^[\u2026.\s]+", "", lines[j]).strip()
                        key = _normalize_zh(line)
                        if len(key) >= 4:
                            glossary.setdefault(key, en)
            i += 1
    return glossary


def rewrite_query(query: str, category: str, glossary: dict[str, str] | None = None) -> str:
    if category == "vocab":
        m = _EN_WORD_IN_QUERY.search(query)
        if m:
            return f"{m.group(1)} 释义"
        return query
    if category == "translation" and glossary:
        qn = _normalize_zh(query)
        hits: list[str] = []
        for key in sorted(glossary, key=len, reverse=True):
            if key and key in qn:
                hits.append(glossary[key])
                if len(hits) >= 2:
                    break
        if hits:
            return query + " " + " ".join(dict.fromkeys(hits))
    return query


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= chunk_size:
            buf = (buf + "\n\n" + p).strip()
            continue
        if buf:
            chunks.append(buf)
            buf = ""
        if len(p) <= chunk_size:
            buf = p
            continue
        step = max(1, chunk_size - overlap)
        for i in range(0, len(p), step):
            piece = p[i : i + chunk_size].strip()
            if piece:
                chunks.append(piece)
            if i + chunk_size >= len(p):
                break
    if buf:
        chunks.append(buf)
    return chunks


def build_chunks(
    doc_dir: Path,
    chunk_size: int,
    overlap: int,
    max_files: int,
    mode: str = "structured",
) -> list[dict]:
    files = sorted(doc_dir.rglob("*.md"))
    if max_files:
        files = files[:max_files]
    chunks: list[dict] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if mode == "structured" and "<table" in text.lower():
            pieces = split_table_entries(text)
            if pieces:
                for idx, piece in enumerate(pieces):
                    chunks.append(
                        {
                            "document_name": path.name,
                            "chunk_id": f"{path.stem}#{idx}",
                            "content": piece,
                        }
                    )
                continue
        if mode == "structured" and any(f in path.name for f in _GLOSSARY_DOC_FRAGS):
            # 词条/对照类文本：按条目（空行分隔的段落）切块，不合并成大块
            pieces = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            for idx, piece in enumerate(pieces):
                chunks.append(
                    {
                        "document_name": path.name,
                        "chunk_id": f"{path.stem}#{idx}",
                        "content": piece,
                    }
                )
            continue
        for idx, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append(
                {
                    "document_name": path.name,
                    "chunk_id": f"{path.stem}#{idx}",
                    "content": piece,
                }
            )
    return chunks


def _lexical_matrix(chunks: list[dict], queries: list[str]) -> tuple[Any, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=1,
        max_features=300_000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform([c["content"] for c in chunks])
    matrix = normalize(matrix)
    qmat = normalize(vectorizer.transform(queries))
    return qmat, matrix


def retrieve_tfidf(chunks: list[dict], queries: list[str], top_k: int) -> list[dict]:
    qmat, matrix = _lexical_matrix(chunks, queries)
    results: list[dict] = []
    for q in qmat:
        scores = (matrix @ q.T).toarray().ravel()
        order = np.argsort(-scores)[:top_k]
        results.append(
            {
                "data": {
                    "chunks": [
                        {
                            "document_name": chunks[i]["document_name"],
                            "content": chunks[i]["content"],
                            "score": float(scores[i]),
                        }
                        for i in order
                    ]
                }
            }
        )
    return results


def _rrf_fuse(
    dense: np.ndarray,
    lex: np.ndarray,
    top_k: int,
    topn: int = 200,
    k: int = 60,
) -> list[int]:
    """对单个 query 的 dense / lex 打分做 RRF 融合，返回 top-k chunk 下标。"""
    d_idx = np.argsort(-dense)[:topn]
    l_idx = np.argsort(-lex)[:topn]
    d_rank = {int(i): j + 1 for j, i in enumerate(d_idx)}
    l_rank = {int(i): j + 1 for j, i in enumerate(l_idx)}
    cand = list(d_rank) + [i for i in l_rank if i not in d_rank]
    score = {
        i: 1.0 / (k + d_rank.get(i, topn + 1)) + 1.0 / (k + l_rank.get(i, topn + 1))
        for i in cand
    }
    return sorted(cand, key=lambda i: -score[i])[:top_k]


def retrieve_hybrid(
    chunks: list[dict],
    queries: list[str],
    top_k: int,
    cache_path: Path,
) -> list[dict]:
    """bge-m3 密集 + TF-IDF 词法双路召回，RRF 融合。"""
    texts = [c["content"] for c in chunks]
    if cache_path.exists():
        doc_emb = np.load(cache_path)
    else:
        doc_emb = encode_bge_m3(texts)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, doc_emb)
    query_emb = encode_bge_m3(queries, batch_size=4)
    dense_all = query_emb @ doc_emb.T
    qmat, cmat = _lexical_matrix(chunks, queries)
    lex_all = (qmat @ cmat.T).toarray()

    results: list[dict] = []
    for dense, lex in zip(dense_all, lex_all):
        order = _rrf_fuse(dense, lex, top_k)
        results.append(
            {
                "data": {
                    "chunks": [
                        {
                            "document_name": chunks[i]["document_name"],
                            "content": chunks[i]["content"],
                            "score": float(dense[i]),
                        }
                        for i in order
                    ]
                }
            }
        )
    return results


def encode_bge_m3(texts: list[str], batch_size: int = 8) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = os.getenv("BGE_M3_MODEL", "BAAI/bge-m3")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bge-m3] loading {model_name} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    model.to(device)
    if device == "cuda":
        model = model.half()
    model.eval()

    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
        vectors.append(hidden.cpu().numpy())
        if start % (batch_size * 10) == 0:
            print(f"[bge-m3] encoded {start + len(batch)}/{len(texts)}", flush=True)
    return np.vstack(vectors)


def retrieve_bge_m3(
    chunks: list[dict],
    queries: list[str],
    top_k: int,
    cache_path: Path,
) -> list[dict]:
    texts = [c["content"] for c in chunks]
    if cache_path.exists():
        doc_emb = np.load(cache_path)
        print(f"[bge-m3] loaded cached embeddings {doc_emb.shape}", flush=True)
    else:
        doc_emb = encode_bge_m3(texts)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, doc_emb)
    query_emb = encode_bge_m3(queries, batch_size=4)
    scores_all = query_emb @ doc_emb.T
    results: list[dict] = []
    for scores in scores_all:
        order = np.argsort(-scores)[:top_k]
        results.append(
            {
                "data": {
                    "chunks": [
                        {
                            "document_name": chunks[i]["document_name"],
                            "content": chunks[i]["content"],
                            "score": float(scores[i]),
                        }
                        for i in order
                    ]
                }
            }
        )
    return results


def summarize(rows: list[dict], backend: str) -> dict:
    n = len(rows)
    summary: dict = {"backend": backend, "n": n}
    for k in KS:
        summary[f"doc@{k}"] = round(sum(r[f"doc@{k}"] for r in rows) / n, 4)
        summary[f"kw@{k}"] = round(sum(r[f"kw@{k}"] for r in rows) / n, 4)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="本地离线检索评测（不依赖 RAGFlow）")
    parser.add_argument("--backend", choices=("tfidf", "bge-m3", "hybrid"), default="hybrid")
    parser.add_argument("--set", default="eval/eval_set.json")
    parser.add_argument("--docs", default="数据集")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--chunking", choices=("naive", "structured"), default="structured")
    parser.add_argument("--rewrite", choices=("none", "basic"), default="basic")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="eval/output")
    args = parser.parse_args()

    questions = json.loads((ROOT / args.set).read_text(encoding="utf-8"))["questions"]
    if args.limit:
        questions = questions[: args.limit]

    glossary = build_translation_glossary(ROOT / args.docs) if args.rewrite == "basic" else {}
    queries = [
        rewrite_query(q["query"], q.get("category", ""), glossary)
        for q in questions
    ]

    print(
        f"[build] chunking docs from {args.docs} (mode={args.chunking}, rewrite={args.rewrite})",
        flush=True,
    )
    chunks = build_chunks(
        ROOT / args.docs, args.chunk_size, args.overlap, args.max_files, mode=args.chunking
    )
    print(f"[build] {len(chunks)} chunks from {len({c['document_name'] for c in chunks})} docs", flush=True)

    t0 = time.perf_counter()
    if args.backend == "tfidf":
        results = retrieve_tfidf(chunks, queries, args.topk)
    elif args.backend == "hybrid":
        cache = ROOT / "eval" / "output" / f"bge_m3_doc_embeddings_{args.chunking}.npy"
        results = retrieve_hybrid(chunks, queries, args.topk, cache)
    else:
        cache = ROOT / "eval" / "output" / f"bge_m3_doc_embeddings_{args.chunking}.npy"
        results = retrieve_bge_m3(chunks, queries, args.topk, cache)
    elapsed = time.perf_counter() - t0

    rows = []
    for q, result in zip(questions, results):
        row = evaluate_item(result, q, KS)
        row["elapsed_s"] = round(elapsed / len(questions), 3)
        rows.append(row)

    summary = summarize(rows, args.backend)
    summary["retrieval_total_s"] = round(elapsed, 2)
    summary["chunks"] = len(chunks)
    summary["docs"] = len({c["document_name"] for c in chunks})
    summary["note"] = "本地离线复现，不是 RAGFlow 线上指标"

    print("\n========== 离线检索结果 ==========")
    for key, val in summary.items():
        print(f"{key}: {val}")
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cat[row["category"]].append(row)
    print("\n按类别 doc@5 / kw@5：")
    for cat, rs in sorted(by_cat.items()):
        d5 = sum(r["doc@5"] for r in rs) / len(rs)
        k5 = sum(r["kw@5"] for r in rs) / len(rs)
        print(f"  {cat:12s} n={len(rs):2d}  doc@5={d5:.2%}  kw@5={k5:.2%}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.backend
    if args.chunking != "naive":
        tag += f"_{args.chunking}"
    if args.rewrite != "none":
        tag += "_rw"
    csv_path = out_dir / f"local_results_{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = out_dir / f"local_summary_{tag}.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入: {csv_path} / {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
