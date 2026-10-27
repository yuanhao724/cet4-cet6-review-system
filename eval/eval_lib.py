"""检索评测的核心打分逻辑（纯函数，无网络依赖，便于单测与复用）。

评测口径（与简历"知识点检索命中率"对齐）：
- doc@k：正确来源文档出现在检索返回的 top-k 个 chunk 的 document_name 中
  （文档级召回率，最接近"命中了知识点所在文档"）。
- kw@k：标准答案关键词出现在 top-k 个 chunk 的内容中
  （答案级命中率，当同一知识点在多个文档中都有合法出处时更公平）。
"""
from __future__ import annotations

from typing import Any


def extract_chunks(search_result: Any) -> list[dict]:
    """从 RAGFlow search 返回结果中提取 chunk 列表。

    兼容两种结构：
    1) {"data": {"chunks": [...]}}
    2) {"data": [{"chunks": [...]}, ...]}
    """
    chunks: list[dict] = []
    if not isinstance(search_result, dict):
        return chunks
    data = search_result.get("data")
    if isinstance(data, dict):
        chunks = data.get("chunks") or []
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                chunks.extend(item.get("chunks") or [])
    return [c for c in chunks if isinstance(c, dict)]


def chunk_content(chunk: dict) -> str:
    return (chunk.get("content") or chunk.get("content_ltks") or "").strip()


def chunk_doc_name(chunk: dict) -> str:
    return (chunk.get("document_name") or chunk.get("doc_name") or "").strip()


def doc_hit(chunks: list[dict], doc_keys: list[str], top_k: int | None = None) -> bool:
    """top-k 个 chunk 中是否有任一 chunk 的文档名命中任一 doc_key。"""
    subset = chunks[:top_k] if top_k is not None else chunks
    for chunk in subset:
        name = chunk_doc_name(chunk)
        for key in doc_keys:
            if key and key in name:
                return True
    return False


def kw_hit(chunks: list[dict], keywords: list[str], top_k: int | None = None) -> bool:
    """top-k 个 chunk 的内容中是否出现任一答案关键词（大小写不敏感）。"""
    subset = chunks[:top_k] if top_k is not None else chunks
    for chunk in subset:
        content = chunk_content(chunk)
        for kw in keywords:
            if kw and kw.lower() in content.lower():
                return True
    return False


def evaluate_item(
    search_result: Any,
    item: dict,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict:
    """计算单题在 k=1/3/5 下的 doc 与 kw 命中情况，返回一行结果。"""
    chunks = extract_chunks(search_result)
    row: dict[str, Any] = {
        "id": item["id"],
        "category": item.get("category", ""),
        "n_retrieved": len(chunks),
    }
    doc_keys = item.get("doc_keys") or []
    keywords = item.get("keywords") or []
    for k in ks:
        row[f"doc@{k}"] = int(doc_hit(chunks, doc_keys, k))
        row[f"kw@{k}"] = int(kw_hit(chunks, keywords, k))
    row["retrieved_docs"] = " | ".join(chunk_doc_name(c) for c in chunks[:5])
    return row
