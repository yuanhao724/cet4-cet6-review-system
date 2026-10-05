import json
import logging
import re
import time
from typing import Any, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents import KBAgent, JudgeAgent, RAGAgent, RAGSynthesizeAgent, ReviewAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("agent_graph")


TaskType = Literal[
    "writing",
    "trans",
    "review",
    "kb_admin",
    "rag_search",
]


class MultiAgentState(TypedDict, total=False):
    task_type: TaskType
    biz_type: str
    query: str
    user_id: int
    content: str
    origin: str
    user_trans: str
    is_admin: str
    rag_context: Any
    llm_raw: Any
    result: Any
    error: Optional[str]
    trace: list[str]


def _append_trace(state: MultiAgentState, agent_name: str) -> list[str]:
    return [*state.get("trace", []), agent_name]


def _loads_llm_json(llm_raw: Any) -> Any:
    """解析LLM返回的JSON，兼容RAGFlow原生格式和OpenAI格式"""
    # RAGFlow原生格式: {"code":0, "data":{"answer":"...", "reference":[...]}}
    if isinstance(llm_raw, dict) and isinstance(llm_raw.get("data"), dict) and "answer" in llm_raw["data"]:
        content = llm_raw["data"]["answer"].strip()
    # OpenAI兼容格式: {"choices":[{"message":{"content":"..."}}]}
    elif isinstance(llm_raw, dict) and "choices" in llm_raw:
        content = llm_raw["choices"][0]["message"]["content"].strip()
    # 错误响应
    elif isinstance(llm_raw, dict) and "error" in llm_raw:
        raise ValueError(f"LLM调用失败: {llm_raw['error']}")
    else:
        raise ValueError(f"无法解析LLM响应格式: {str(llm_raw)[:200]}")

    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.S)
        if match:
            return json.loads(match.group(0))
        raise


def _extract_llm_content(llm_raw: Any) -> str:
    """从LLM响应中提取纯文本内容（用于非JSON场景）"""
    if isinstance(llm_raw, dict) and isinstance(llm_raw.get("data"), dict) and "answer" in llm_raw["data"]:
        return llm_raw["data"]["answer"].strip()
    elif isinstance(llm_raw, dict) and "choices" in llm_raw:
        return llm_raw["choices"][0]["message"]["content"].strip()
    elif isinstance(llm_raw, dict) and "error" in llm_raw:
        return f"[AI回答生成失败] {llm_raw['error']}"
    return str(llm_raw)


def dispatch_node(state: MultiAgentState) -> MultiAgentState:
    return {"trace": _append_trace(state, "Dispatcher")}


def _format_rag_search_result(rag_context: dict) -> str:
    """将RAGFlow原始检索结果格式化为可读文本"""
    MAX_RESULTS = 5       # 最多展示条数
    MAX_CONTENT_LEN = 500 # 每条内容最多字符数

    kb_results = rag_context.get("kb_results", [])
    if not kb_results:
        return "未检索到相关内容"

    lines = []
    chunk_idx = 1
    for kb_res in kb_results:
        if isinstance(kb_res, dict) and kb_res.get("error"):
            lines.append(f"[检索异常] {kb_res['error']}")
            continue
        if not isinstance(kb_res, dict):
            continue

        data = kb_res.get("data")
        if data is None:
            continue

        all_chunks = []
        if isinstance(data, dict):
            all_chunks = data.get("chunks", [])
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    all_chunks.extend(item.get("chunks", []))

        for chunk in all_chunks:
            if chunk_idx > MAX_RESULTS:
                break
            if not isinstance(chunk, dict):
                continue

            content = chunk.get("content", "") or chunk.get("content_ltks", "")
            content = content.strip()
            if not content:
                continue

            # 清理 content_ltks 格式: "word1 p p 中文1 p p word2 p p 中文2..."
            # 转为每行一对: "word1 → 中文1\nword2 → 中文2"
            if chunk.get("content_ltks") and not chunk.get("content"):
                # 去掉开头的 "p "
                content = re.sub(r"^\s*p\s+", "", content)
                # " p p " → 换行
                content = re.sub(r"\s*p\s+p\s*", "\n", content)

            doc_name = chunk.get("document_name") or chunk.get("doc_name") or ""
            similarity = chunk.get("similarity") or chunk.get("score") or 0

            # 截断过长内容
            truncated = len(content) > MAX_CONTENT_LEN
            if truncated:
                content = content[:MAX_CONTENT_LEN] + "..."

            header = f"【结果 {chunk_idx}】"
            if similarity:
                header += f" 相关度: {float(similarity):.2f}"
            if doc_name:
                header += f" | 来源: {doc_name}"
            if truncated:
                header += " (内容已截断)"

            lines.append(header)
            lines.append(content)
            lines.append("")
            chunk_idx += 1

        if chunk_idx > MAX_RESULTS:
            lines.append(f"... 仅展示前 {MAX_RESULTS} 条结果，其余已省略")
            break

    if not lines:
        return "未检索到相关内容"
    return "\n".join(lines)


def rag_node(state: MultiAgentState) -> MultiAgentState:
    biz_type = state.get("biz_type") or state["task_type"]
    query = state.get("query") or state.get("content") or state.get("origin") or ""
    logger.info(f"[RAG检索开始] biz_type={biz_type}, query_len={len(query)}")

    t0 = time.time()
    try:
        rag_context = RAGAgent.route_search(biz_type, query)
    except Exception as exc:
        logger.exception(f"[RAG检索异常] {exc}")
        rag_context = {"kb_results": [{"error": f"检索节点异常: {str(exc)}"}]}

    elapsed = time.time() - t0
    # 统计成功/失败数
    kb_results = rag_context.get("kb_results", []) if isinstance(rag_context, dict) else []
    ok_count = sum(1 for r in kb_results if isinstance(r, dict) and "error" not in r)
    err_count = len(kb_results) - ok_count
    logger.info(f"[RAG检索完成] 耗时={elapsed:.2f}s, 成功={ok_count}, 失败={err_count}")

    if state["task_type"] == "rag_search":
        result = {"text": _format_rag_search_result(rag_context), "raw": rag_context}
    else:
        result = state.get("result")

    return {
        "biz_type": biz_type,
        "query": query,
        "rag_context": rag_context,
        "result": result,
        "trace": _append_trace(state, "RAGAgent"),
    }


def rag_synthesize_node(state: MultiAgentState) -> MultiAgentState:
    """RAG检索后AI整合回答节点 —— 基于检索结果生成自然语言回答"""
    query = state.get("query") or ""
    rag_context = state.get("rag_context") or {}

    # 检查检索结果是否为空
    kb_results = rag_context.get("kb_results", []) if isinstance(rag_context, dict) else []
    has_error = any(isinstance(r, dict) and "error" in r for r in kb_results)
    has_data = any(
        isinstance(r, dict) and r.get("data") and not r.get("error")
        for r in kb_results
    )

    if has_error and not has_data:
        error_msgs = [r["error"] for r in kb_results if isinstance(r, dict) and "error" in r]
        err_text = "; ".join(error_msgs)
        logger.warning(f"[AI整合回答跳过] 检索全部失败: {err_text}")
        prev_result = state.get("result") or {}
        if isinstance(prev_result, dict):
            prev_result["ai_answer"] = f"知识库检索失败：{err_text}。请检查RAGFlow服务是否正常运行、知识库是否已导入数据。"
        return {
            "llm_raw": {"error": err_text},
            "result": prev_result,
            "trace": _append_trace(state, "RAGSynthesizeAgent"),
        }

    logger.info(f"[AI整合回答开始] query_len={len(query)}, kb_ok={has_data}, kb_err={has_error}")

    t0 = time.time()
    try:
        llm_raw = RAGSynthesizeAgent.synthesize_answer(query, rag_context)
        ai_answer = _extract_llm_content(llm_raw)
        elapsed = time.time() - t0
        logger.info(f"[AI整合回答完成] 耗时={elapsed:.2f}s, answer_len={len(ai_answer)}")

        # 检测是否是LLM错误响应
        if isinstance(llm_raw, dict) and "error" in llm_raw:
            logger.warning(f"[AI整合回答] LLM返回错误: {llm_raw['error'][:200]}")
            ai_answer = f"AI回答生成失败：{llm_raw['error']}\n\n请尝试：\n1. 缩短查询内容\n2. 检查RAGFlow中Chat Assistant的LLM模型配置\n3. 确认DeepSeek API是否正常"

    except ValueError as exc:
        # _extract_llm_content 中解析失败
        logger.exception(f"[AI整合回答解析异常] {exc}")
        llm_raw = {"error": str(exc)}
        ai_answer = f"[AI回答解析失败] {exc}"
    except Exception as exc:
        logger.exception(f"[AI整合回答异常] {exc}")
        llm_raw = {"error": str(exc)}
        ai_answer = f"[AI回答生成失败] {exc}\n\n可能原因：LLM调用超时或RAGFlow服务异常，请稍后重试。"

    # 保留原有检索结果，追加AI整合回答
    prev_result = state.get("result") or {}
    if isinstance(prev_result, dict):
        prev_result["ai_answer"] = ai_answer
    else:
        prev_result = {"text": str(prev_result), "ai_answer": ai_answer}

    return {
        "llm_raw": llm_raw,
        "result": prev_result,
        "trace": _append_trace(state, "RAGSynthesizeAgent"),
    }


def judge_node(state: MultiAgentState) -> MultiAgentState:
    if state["task_type"] == "writing":
        llm_raw = JudgeAgent.judge_writing(state.get("content", ""), state.get("rag_context"))
    elif state["task_type"] == "trans":
        llm_raw = JudgeAgent.judge_trans(
            state.get("origin", ""),
            state.get("user_trans", ""),
            state.get("rag_context"),
        )
    else:
        return {
            "error": "JudgeAgent仅支持作文和翻译批改任务",
            "trace": _append_trace(state, "JudgeAgent"),
        }

    result = _loads_llm_json(llm_raw)
    return {
        "llm_raw": llm_raw,
        "result": result,
        "trace": _append_trace(state, "JudgeAgent"),
    }


def review_node(state: MultiAgentState) -> MultiAgentState:
    result = ReviewAgent.plan_review(state["user_id"])
    return {
        "result": result,
        "trace": _append_trace(state, "ReviewAgent"),
    }


def kb_node(state: MultiAgentState) -> MultiAgentState:
    if state.get("is_admin") != "1":
        return {
            "result": {"msg": "权限不足，仅管理员可查看知识库运维建议"},
            "trace": _append_trace(state, "KBAgent"),
        }
    return {
        "result": {"suggest": KBAgent.stat_hot_error()},
        "trace": _append_trace(state, "KBAgent"),
    }


def route_after_dispatch(state: MultiAgentState) -> str:
    route_map = {
        "writing": "rag",
        "trans": "rag",
        "review": "review",
        "kb_admin": "kb",
        "rag_search": "rag",
    }
    return route_map.get(state["task_type"], "end")


def route_after_rag(state: MultiAgentState) -> str:
    if state["task_type"] in {"writing", "trans"}:
        return "judge"
    if state["task_type"] == "rag_search":
        return "synthesize"
    return "end"


def build_multi_agent_graph():
    graph = StateGraph(MultiAgentState)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("rag", rag_node)
    graph.add_node("judge", judge_node)
    graph.add_node("synthesize", rag_synthesize_node)
    graph.add_node("review", review_node)
    graph.add_node("kb", kb_node)

    graph.set_entry_point("dispatch")
    graph.add_conditional_edges(
        "dispatch",
        route_after_dispatch,
        {
            "rag": "rag",
            "review": "review",
            "kb": "kb",
            "end": END,
        },
    )
    graph.add_conditional_edges("rag", route_after_rag, {"judge": "judge", "synthesize": "synthesize", "end": END})
    graph.add_edge("judge", END)
    graph.add_edge("synthesize", END)
    graph.add_edge("review", END)
    graph.add_edge("kb", END)
    return graph.compile()


multi_agent_graph = build_multi_agent_graph()
