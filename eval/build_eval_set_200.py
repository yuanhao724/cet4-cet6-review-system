# -*- coding: utf-8 -*-
"""从知识库真实文档自动构建 200+ 题检索评测集（词汇/翻译/写作三子集）。

产出 eval/eval_set_200.json，schema 与 eval_set.json 一致：
  id, category, query, answer, doc_keys, keywords
题目、答案、关键词均取自 数据集/ 目录真实内容；doc_keys 为来源文档名的稳定片段。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "数据集"

_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S | re.I)
_CELL = re.compile(r"<(?:td|th)\b[^>]*>(.*?)</(?:td|th)>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_WORD_LIKE = re.compile(r"^[A-Za-z][A-Za-z\-'. ]{0,39}$")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]{2,}")


def _cell_text(html: str) -> str:
    html = (
        html.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
    )
    return _TAG.sub(" ", html).strip()


def _read(name_glob: str, *subdirs: str) -> Path:
    matches = list(DATA.joinpath(*subdirs).glob(name_glob))
    if not matches:
        raise FileNotFoundError(f"no doc: {subdirs}/{name_glob}")
    return sorted(matches)[0]


def _frag(path: Path) -> str:
    name = path.name
    if "四级核心单词" in name:
        return "四级核心单词"
    if "六级核心单词" in name:
        return "六级核心单词"
    if "四级词汇完整版" in name:
        return "四级词汇完整版"
    if "六级词汇完整版" in name:
        return "六级词汇完整版"
    if "翻译万能固定句式" in name:
        return "翻译万能固定句式"
    if "翻译高频扣分" in name:
        return "翻译高频扣分"
    if "翻译分话题高频词汇对照表" in name:
        return "翻译分话题高频词汇对照表"
    if "高低级词汇替换" in name:
        return "高低级词汇替换"
    if "高分万能句式库" in name:
        return "高分万能句式"
    if "四级应用文" in name:
        return "四级应用文"
    if "六级议论文" in name:
        return "六级议论文"
    return name


def extract_vocab() -> list[tuple[str, str, str]]:
    """(word, meaning, doc_fragment) from 核心单词 HTML tables."""
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for glob, subdir in [("*四级核心单词*.md", "单词"), ("*六级核心单词*.md", "单词")]:
        path = _read(glob, subdir)
        text = path.read_text(encoding="utf-8", errors="ignore")
        frag = _frag(path)
        for row in _ROW.finditer(text):
            cells = [c for c in (_cell_text(c) for c in _CELL.findall(row.group(1))) if c]
            if len(cells) < 2 or any(c in ("序号", "单词", "注音", "释义") for c in cells):
                continue
            word = next((c for c in cells if _WORD_LIKE.match(c)), None)
            cjk = [c for c in cells if _CJK.search(c)]
            meaning = max(cjk, key=len) if cjk else None
            if not word or not meaning:
                continue
            wl = word.lower()
            if wl in seen:
                continue
            seen.add(wl)
            out.append((word, meaning, frag))
    return out


def extract_translation() -> list[tuple[str, str, str]]:
    """(query, answer, doc_fragment) from 对照表 + 高频扣分 + 万能句式。"""
    out: list[tuple[str, str, str]] = []
    # 对照表：中文 english
    path = _read("*翻译分话题高频词汇对照表*.md", "翻译")
    text = path.read_text(encoding="utf-8", errors="ignore")
    frag = _frag(path)
    pat = re.compile(r"^\s*([\u4e00-\u9fff][\u4e00-\u9fff（）·、]*?)\s+([A-Za-z][A-Za-z\-() ]+?)(?:\s*[｜|]|\s*$)")
    for line in text.splitlines():
        m = pat.match(line.strip())
        if m and len(m.group(1)) >= 2:
            out.append((f"“{m.group(1)}”的正确英文翻译是什么？", m.group(2).strip(), frag))
    # 高频扣分：错误：X｜正确：Y
    path = _read("*翻译高频扣分*.md", "翻译")
    text = path.read_text(encoding="utf-8", errors="ignore")
    frag = _frag(path)
    pat = re.compile(r"错误[：:]\s*(.+?)\s*[｜|]\s*正确[：:]\s*(.+)")
    for line in text.splitlines():
        m = pat.search(line)
        if m:
            wrong = m.group(1).strip()
            right = re.sub(r"（[^）]*）", "", m.group(2)).strip()
            if wrong and right:
                out.append((f"四六级翻译中「{wrong}」的正确英文表达是什么？", right, frag))
    # 万能句式：中文行 + 下一行英文
    path = _read("*翻译万能固定句式*.md", "翻译")
    text = path.read_text(encoding="utf-8", errors="ignore")
    frag = _frag(path)
    lines = [ln.strip() for ln in text.splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i]
        has_cjk = bool(_CJK.search(line))
        has_latin = bool(re.search(r"[A-Za-z]{2,}", line))
        if has_cjk and not has_latin and not re.match(r"^[一二三四五六七八九十]+、", line) and "、" not in line[:4]:
            j = i + 1
            while j < len(lines) and not lines[j]:
                j += 1
            if j < len(lines) and re.search(r"[A-Za-z]{2,}", lines[j]):
                zh = re.sub(r"^[\u2026.\s]+", "", line).strip("，。；: ")
                en = re.sub(r"^[\u2026.\s]+", "", lines[j]).strip()
                if zh and en:
                    out.append((f"翻译中“{zh}”对应的英文表达是什么？", en, frag))
        i += 1
    return out


def extract_writing() -> list[tuple[str, str, str]]:
    """(query, answer, doc_fragment) from 替换表 + 应用文模板 + 议论文模板 + 万能句式库。"""
    out: list[tuple[str, str, str]] = []
    # 高低级词汇替换
    path = _read("*高低级词汇替换表*.md", "写作")
    text = path.read_text(encoding="utf-8", errors="ignore")
    frag = _frag(path)
    for line in text.splitlines():
        m = re.match(r"^\s*([a-z]+)\s*→\s*(.+)$", line.strip())
        if m:
            word, repl = m.group(1), m.group(2).strip()
            out.append((f"写作中 {word} 有哪些高分替换表达？", repl, frag))
    # 应用文模板：体裁 → 开头句
    path = _read("*四级应用文模版*.md", "写作")
    text = path.read_text(encoding="utf-8", errors="ignore")
    frag = _frag(path)
    genre = None
    for line in text.splitlines():
        m = re.search(r"体裁[：:]\s*([^｜\s]+)", line)
        if m:
            genre = m.group(1)
            continue
        s = line.strip()
        if genre and re.match(r"^[A-Z][A-Za-z ,'-]+[.?!]$", s):
            kw = " ".join(s.split()[:5]).rstrip(".,;")
            out.append((f"四级{genre}的开头套句是什么？", s, frag))
            genre = None
    # 六级议论文模板：结构 → 开头句
    path = _read("*六级议论文*.md", "写作")
    text = path.read_text(encoding="utf-8", errors="ignore")
    frag = _frag(path)
    struct = None
    for line in text.splitlines():
        m = re.search(r"(现象解释型|观点对比型|问题解决型)", line)
        if m:
            struct = m.group(1)
            continue
        s = line.strip()
        if struct and re.match(r"^Nowadays,", s):
            out.append((f"六级议论文“{struct}”结构的开头句式是什么？", s, frag))
            struct = None
    # 高分万能句式库：分类 → 句式
    path = _read("*高分万能句式库*.md", "写作")
    text = path.read_text(encoding="utf-8", errors="ignore")
    frag = _frag(path)
    section = None
    for line in text.splitlines():
        m = re.search(r"写作万能句[｜|]([^】]+)】", line)
        if m:
            section = m.group(1)
            continue
        s = line.strip()
        if section and re.match(r"^[A-Z][A-Za-z ,'-]+", s):
            out.append((f"四六级作文「{section}」的常用句式是什么？", s, frag))
            section = None
    return out


def _pick(items: list, n: int) -> list:
    if len(items) <= n:
        return items
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


def main() -> int:
    vocab = extract_vocab()
    trans = extract_translation()
    write = extract_writing()
    print(f"extracted: vocab={len(vocab)} translation={len(trans)} writing={len(write)}")

    # 分层抽样，保证三子集均衡且总量 200+
    v = _pick(vocab, 120)
    t = _pick(trans, 70)
    w = _pick(write, 35)

    questions = []
    for i, (word, meaning, frag) in enumerate(v):
        kw = _CJK_RUN.search(meaning)
        questions.append({
            "id": len(questions) + 1,
            "category": "vocab",
            "query": f"四六级核心词汇中，单词 {word} 的释义是什么？",
            "answer": meaning,
            "doc_keys": [frag],
            "keywords": [kw.group(0) if kw else meaning],
        })
    for query, answer, frag in t:
        questions.append({
            "id": len(questions) + 1,
            "category": "translation",
            "query": query,
            "answer": answer,
            "doc_keys": [frag],
            "keywords": [answer.strip(" .…")],
        })
    for query, answer, frag in w:
        questions.append({
            "id": len(questions) + 1,
            "category": "writing",
            "query": query,
            "answer": answer,
            "doc_keys": [frag],
            "keywords": [answer.strip(" .…")],
        })

    meta = {
        "name": "四六级知识库检索命中率评测集（200+ 扩展版）",
        "version": 2,
        "n_questions": len(questions),
        "top_k_eval": [1, 3, 5],
        "hit_definition": {
            "doc@k": "正确来源文档出现在 top-k chunk 的 document_name 中（文档级召回率）",
            "kw@k": "标准答案关键词出现在 top-k chunk 内容中（答案级命中率，主口径）",
        },
        "note": "题目与答案自动抽取自 数据集/ 目录真实文档（词汇/翻译/写作三子集），关键词为答案的首个连续片段。",
    }
    out_path = ROOT / "eval" / "eval_set_200.json"
    out_path.write_text(
        json.dumps({"meta": meta, "questions": questions}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    from collections import Counter
    print("final:", Counter(q["category"] for q in questions), "total", len(questions))
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
