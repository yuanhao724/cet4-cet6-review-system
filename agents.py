import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pymysql
import requests
from dotenv import load_dotenv

from db import get_db_connection
from ragflow_api import search_kb

load_dotenv()

# 检索线程池 —— 避免串行调用堆叠超时
_search_executor = ThreadPoolExecutor(max_workers=4)


def llm_call(prompt: str):
    """LLM统一调用封装 —— 优先使用 no-KB 助手，避免 RAGFlow Chat Assistant 内部冗余KB检索导致超时。

    架构说明：
    - 上游 rag_node 已经完成 KB 检索，结果已拼入 prompt
    - 此处只需纯 LLM 调用，不应再触发 KB 检索
    - 因此使用 RAGFLOW_CHAT_ID_LLM（不绑定KB的助手），而非 RAGFLOW_CHAT_ID（带KB的助手）

    回退策略：如果 RAGFLOW_CHAT_ID_LLM 未配置，则回退到 RAGFLOW_CHAT_ID（带KB，可能超时），
    并输出日志提示用户创建 no-KB 助手。
    """
    base_url = os.getenv("RAGFLOW_BASE_URL")
    api_key = os.getenv("RAGFLOW_API_KEY")
    model_name = os.getenv("LLM_MODEL_NAME", "deepseek-v4-flash@shiki@DeepSeek")

    # 优先使用 no-KB 助手（避免冗余检索 + 超时）
    chat_id_llm = os.getenv("RAGFLOW_CHAT_ID_LLM", "").strip()
    chat_id_kb = os.getenv("RAGFLOW_CHAT_ID", "").strip()

    if not api_key:
        return {"error": "RAGFLOW_API_KEY 未在 .env 中配置"}

    if chat_id_llm:
        chat_id = chat_id_llm
    elif chat_id_kb:
        import logging
        logging.getLogger("agents").warning(
            "RAGFLOW_CHAT_ID_LLM 未配置，回退到带KB的Chat Assistant（可能导致长prompt超时）。"
            "建议在RAGFlow控制台创建一个不绑定KB的Chat Assistant，将其ID填入 .env 的 RAGFLOW_CHAT_ID_LLM。"
        )
        chat_id = chat_id_kb
    else:
        return {"error": "RAGFLOW_CHAT_ID 和 RAGFLOW_CHAT_ID_LLM 都未配置，请在RAGFlow控制台创建Chat Assistant"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # OpenAI 兼容端点：/api/v1/chats_openai/{chat_id}/chat/completions
    url = f"{base_url}/api/v1/chats_openai/{chat_id}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=(10, 90))
        if resp.status_code == 200:
            result = resp.json()
            # OpenAI 兼容格式: {"choices": [{"message": {"content": "..."}}]}
            choices = result.get("choices", [])
            if choices and choices[0].get("message", {}).get("content"):
                return result
            return {
                "error": (
                    f"LLM返回空内容。"
                    f"请检查RAGFlow中Chat Assistant是否已配置LLM模型"
                ),
                "_raw": result,
            }
        return {"error": f"LLM调用失败 HTTP {resp.status_code}: {resp.text[:300]}"}
    except requests.exceptions.ConnectionError:
        return {"error": f"无法连接RAGFlow服务 ({base_url})，请确认Docker容器正在运行"}
    except requests.exceptions.Timeout:
        return {"error": "LLM调用请求超时（10s连接 / 90s读取）。建议检查prompt长度或创建no-KB助手。"}
    except Exception as e:
        return {"error": f"LLM调用异常: {str(e)}"}


def _format_rag_context_for_llm(rag_context, max_chunks: int = 5, max_chunk_len: int = 400) -> str:
    """将RAG检索结果格式化为LLM友好的紧凑文本，避免prompt过长导致超时。

    与 agent_graph._format_rag_search_result 不同，此函数输出更紧凑（无表头装饰），
    优先使用 content_ltks（词汇表格式），适合喂给LLM。
    """
    import re

    kb_results = rag_context.get("kb_results", []) if isinstance(rag_context, dict) else []
    if not kb_results:
        return "未检索到相关内容"

    lines = []
    chunk_idx = 0
    for kb_res in kb_results:
        if not isinstance(kb_res, dict):
            continue
        if kb_res.get("error"):
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
            if chunk_idx >= max_chunks:
                break
            if not isinstance(chunk, dict):
                continue

            content = chunk.get("content", "") or chunk.get("content_ltks", "")
            content = content.strip()
            if not content:
                continue

            # 清理 content_ltks 格式: "word1 p p 中文1 p p word2 p p 中文2..."
            if chunk.get("content_ltks") and not chunk.get("content"):
                content = re.sub(r"^\s*p\s+", "", content)
                content = re.sub(r"\s*p\s+p\s*", "\n", content)

            doc_name = chunk.get("document_name") or chunk.get("doc_name") or ""
            similarity = chunk.get("similarity") or chunk.get("score") or 0

            # 截断过长内容
            if len(content) > max_chunk_len:
                content = content[:max_chunk_len] + "..."

            # 紧凑格式：一行来源 + 内容
            header = f"[来源: {doc_name}]" if doc_name else ""
            if similarity and doc_name:
                header += f" 相关度: {float(similarity):.2f}"
            elif similarity:
                header = f"[相关度: {float(similarity):.2f}]"

            if header:
                lines.append(header)
            lines.append(content)
            lines.append("")
            chunk_idx += 1

        if chunk_idx >= max_chunks:
            break

    if not lines:
        return "未检索到相关内容"
    return "\n".join(lines)


class RAGSynthesizeAgent:
    @staticmethod
    def synthesize_answer(query: str, rag_context):
        """基于RAG检索结果，调用LLM整合生成自然语言回答。

        重要：使用 _format_rag_context_for_llm 对检索结果做截断（最多5条、每条400字符），
        避免将原始RAGFlow全量JSON塞入prompt导致token爆炸 → LLM超时。
        """
        context_str = _format_rag_context_for_llm(rag_context)

        # 二次防护：如果格式化后仍然过长，进一步截断
        MAX_CONTEXT_CHARS = 2500
        if len(context_str) > MAX_CONTEXT_CHARS:
            context_str = context_str[:MAX_CONTEXT_CHARS] + "\n...(检索结果过长已截断)"

        prompt = f"""
你是四六级英语备考AI助手，请严格基于下方「知识库检索参考资料」，对用户的问题进行专业、完整、易懂的回答。

要求：
1. 必须结合参考资料中的知识点来回答，不要脱离资料凭空生成；
2. 如果资料中有相关例句、模板、技巧、评分标准等，请重点引用并加以解释；
3. 回答结构清晰、条理分明，使用分点或分段让用户容易理解；
4. 如果参考资料不足以充分回答问题，请如实说明，不要编造；
5. 只返回纯文本回答，不要JSON格式，不要markdown代码块。

用户问题：
{query}

知识库检索参考资料：
{context_str}
"""
        return llm_call(prompt)


class OCRAgent:
    _reader = None  # EasyOCR reader 单例
    _easyocr_disabled = False  # 永久禁用标记（如numpy版本不兼容）

    @classmethod
    def _get_reader(cls):
        """延迟初始化 EasyOCR reader，任意异常都降级到LLM回退"""
        if cls._easyocr_disabled:
            return None
        if cls._reader is None:
            try:
                import easyocr
                cls._reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
            except BaseException as e:
                # 包括 ImportError(numpy不兼容)、OSError、SystemError 等所有异常
                # 发生任何错误都永久禁用 easyocr，走 LLM fallback
                cls._easyocr_disabled = True
                cls._reader = None
                import logging
                logging.getLogger("agents").warning(
                    f"EasyOCR初始化失败，已禁用。将使用LLM视觉识别回退。错误: {e}"
                )
                return None
        return cls._reader

    @staticmethod
    def extract_text(img_base64: str | None):
        """OCR识别图片中的文字（中英文）。

        使用 EasyOCR 作为识别引擎，若未安装则回退到基于LLM的base64视觉识别。

        Args:
            img_base64: base64编码的图片字符串（可含 data:image/...;base64, 前缀）

        Returns:
            {"text": "识别出的文字内容", "engine": "easyocr"|"llm_fallback"|"none"}
        """
        if not img_base64:
            return {"text": "", "engine": "none"}

        # 去除可能的 data URL 前缀
        clean_base64 = img_base64
        if "," in img_base64 and img_base64.startswith("data:"):
            clean_base64 = img_base64.split(",", 1)[1]

        import base64

        # 方案1：EasyOCR（优先，本地识别，无网络依赖）
        reader = OCRAgent._get_reader()
        if reader is not None:
            try:
                import numpy as np
                from PIL import Image
                import io

                img_bytes = base64.b64decode(clean_base64)
                img = Image.open(io.BytesIO(img_bytes))
                img_np = np.array(img)

                results = reader.readtext(img_np)
                text_lines = [item[1] for item in results if item[2] >= 0.3]  # 置信度>=0.3
                text = "\n".join(text_lines)
                return {"text": text, "engine": "easyocr"}
            except Exception as e:
                # EasyOCR 识别失败，尝试LLM fallback
                pass

        # 方案2：LLM视觉识别回退（DeepSeek 支持图片理解）
        try:
            llm_result = _ocr_via_llm(clean_base64)
            if llm_result and not llm_result.startswith("[OCR"):
                return {"text": llm_result, "engine": "llm_fallback"}
        except Exception:
            pass

        return {
            "text": "",
            "engine": "none",
            "hint": "OCR未识别到文字。请确保：1) 图片清晰 2) 安装EasyOCR: pip install easyocr 3) 或配置支持视觉的LLM模型",
        }


def _ocr_via_llm(img_base64: str) -> str:
    """通过LLM视觉能力识别图片文字（DeepSeek v4 等支持图片输入的模型）"""
    base_url = os.getenv("RAGFLOW_BASE_URL")
    api_key = os.getenv("RAGFLOW_API_KEY")
    chat_id_llm = os.getenv("RAGFLOW_CHAT_ID_LLM", "").strip()
    model_name = os.getenv("LLM_MODEL_NAME", "deepseek-v4-flash@shiki@DeepSeek")

    if not base_url or not api_key or not chat_id_llm:
        return "[OCR] LLM配置不完整，无法使用视觉识别"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = f"{base_url}/api/v1/chats_openai/{chat_id_llm}/chat/completions"

    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "请直接识别并输出图片中的所有文字内容，不要添加任何额外解释或格式。如果是英文，原样输出；如果是中文，原样输出。逐行输出，保持原文排版。"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                }
            ]
        }],
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=(10, 60))
        if resp.status_code == 200:
            result = resp.json()
            choices = result.get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            return content.strip()
        return f"[OCR] LLM调用失败 HTTP {resp.status_code}"
    except requests.exceptions.Timeout:
        return "[OCR] LLM视觉识别超时"
    except Exception as e:
        return f"[OCR] 异常: {str(e)}"


class RAGAgent:
    @staticmethod
    def route_search(biz_type: str, query: str):
        """智能路由分发 —— 所有业务共用唯一知识库，仅检索query不同。
        对 writing/trans 任务，两个检索query并行执行，避免串行等待堆叠超时。"""
        if biz_type == "writing":
            # 作文批改：并行检索写作范文 + 评分标准（不再串行等待）
            futures = {
                _search_executor.submit(search_kb, query): "query",
                _search_executor.submit(search_kb, "四六级作文五档评分标准、扣分细则、高分范文"): "rule",
            }
            results = {}
            for future in as_completed(futures):
                label = futures[future]
                try:
                    results[label] = future.result(timeout=25)
                except Exception as e:
                    results[label] = {"error": f"并行检索异常({label}): {str(e)}"}
            # 保持与原有顺序一致：query 结果在前，rule 在后
            return {"kb_results": [results.get("query", {}), results.get("rule", {})]}

        if biz_type == "trans":
            # 翻译批改：并行检索翻译素材 + 评分标准
            futures = {
                _search_executor.submit(search_kb, query): "query",
                _search_executor.submit(search_kb, "四六级翻译五档评分标准、扣分细则、参考译文"): "rule",
            }
            results = {}
            for future in as_completed(futures):
                label = futures[future]
                try:
                    results[label] = future.result(timeout=25)
                except Exception as e:
                    results[label] = {"error": f"并行检索异常({label}): {str(e)}"}
            return {"kb_results": [results.get("query", {}), results.get("rule", {})]}

        if biz_type == "error_analysis":
            # 错题分析：并行检索语法 + 解题方法
            futures = {
                _search_executor.submit(search_kb, query): "query",
                _search_executor.submit(search_kb, "对应题型解题步骤与避坑方法"): "method",
            }
            results = {}
            for future in as_completed(futures):
                label = futures[future]
                try:
                    results[label] = future.result(timeout=25)
                except Exception as e:
                    results[label] = {"error": f"并行检索异常({label}): {str(e)}"}
            return {"kb_results": [results.get("query", {}), results.get("method", {})]}

        # 通用检索：vocab / grammar / paper / rule / rag_search 等（单次检索，无需并行）
        return {"kb_results": [search_kb(query)]}


class JudgeAgent:
    @staticmethod
    def judge_writing(content: str, rag_context):
        """作文批改打分、点评、生成修改稿"""
        # 使用紧凑格式化避免prompt过长导致LLM超时（与synthesize_answer同样的问题）
        context_str = _format_rag_context_for_llm(rag_context, max_chunks=4, max_chunk_len=350)
        if len(context_str) > 2000:
            context_str = context_str[:2000] + "\n...(已截断)"

        prompt = f"""
你是四六级英语阅卷老师，严格参考下方知识库评分规则与写作素材，对学生作文打分（满分15）。
要求：
1. 给出精确总分；
2. 逐条说明语病、低级词汇、逻辑结构、语法错误问题；
3. 输出润色修改后的完整作文；
4. 最终仅返回JSON字符串，格式严格如下，不要额外解释文字：
{{"score": 分数数字,"comment":"详细评语","revise":"修改后完整作文"}}

参考知识库内容：
{context_str}
学生作文原文：
{content}
"""
        return llm_call(prompt)

    @staticmethod
    def judge_trans(origin: str, user_trans: str, rag_context):
        """翻译批改打分、点评、优化译文"""
        context_str = _format_rag_context_for_llm(rag_context, max_chunks=4, max_chunk_len=350)
        if len(context_str) > 2000:
            context_str = context_str[:2000] + "\n...(已截断)"

        prompt = f"""
你是四六级翻译阅卷老师，严格依据下方评分细则与翻译素材批改译文，满分15分。
要求：
1. 给出精确得分；
2. 指出漏译、中式英语、用词错误、语法问题；
3. 给出通顺地道优化版译文；
4. 只返回JSON字符串，格式如下，禁止多余文字：
{{"score":分数数字,"suggest":"点评分析","better_trans":"优化参考译文"}}

参考资料：
{context_str}
中文原文：
{origin}
学生翻译译文：
{user_trans}
"""
        return llm_call(prompt)


class ReviewAgent:
    @staticmethod
    def plan_review(user_id):
        """基于艾宾浩斯周期生成待复习错题任务。

        增强功能：
        1. 自动扫描 writing_submit / trans_submit 中 score<=9 但未创建 error_record 的条目并补建
        2. 通过 source_type / source_id 关联原始作文/翻译内容，让复习任务可展示具体内容
        """
        conn = get_db_connection()
        cur = conn.cursor(cursor=pymysql.cursors.DictCursor)

        # ===== 步骤1：补建漏掉的低分作文/翻译错题记录 =====
        synced_count = 0

        # 扫描 writing_submit 中 score<=9 但无对应 error_record
        cur.execute("""
            SELECT ws.id, ws.user_id
            FROM writing_submit ws
            WHERE ws.user_id = %s AND ws.score <= 9
            AND NOT EXISTS (
                SELECT 1 FROM error_record er
                WHERE er.source_type = 'writing' AND er.source_id = ws.id
            )
        """, (user_id,))
        for w in cur.fetchall():
            cur.execute("""
                INSERT INTO error_record(user_id, error_type, source_type, source_id, first_wrong)
                VALUES(%s, 'method', 'writing', %s, NOW())
            """, (user_id, w['id']))
            synced_count += 1

        # 扫描 trans_submit 中 score<=9 但无对应 error_record
        cur.execute("""
            SELECT ts.id, ts.user_id
            FROM trans_submit ts
            WHERE ts.user_id = %s AND ts.score <= 9
            AND NOT EXISTS (
                SELECT 1 FROM error_record er
                WHERE er.source_type = 'trans' AND er.source_id = ts.id
            )
        """, (user_id,))
        for t in cur.fetchall():
            cur.execute("""
                INSERT INTO error_record(user_id, error_type, source_type, source_id, first_wrong)
                VALUES(%s, 'grammar', 'trans', %s, NOW())
            """, (user_id, t['id']))
            synced_count += 1

        conn.commit()

        # ===== 步骤2：查询所有未完成错题，LEFT JOIN 关联原始内容 =====
        cur.execute("""
            SELECT er.*,
                   ws.content AS writing_content, ws.score AS writing_score,
                   ws.comment AS writing_comment, ws.revise_content AS writing_revise,
                   ts.origin_text AS trans_origin, ts.user_trans AS trans_user_trans,
                   ts.score AS trans_score, ts.suggest AS trans_suggest
            FROM error_record er
            LEFT JOIN writing_submit ws
                ON er.source_type = 'writing' AND er.source_id = ws.id
            LEFT JOIN trans_submit ts
                ON er.source_type = 'trans' AND er.source_id = ts.id
            WHERE er.user_id = %s AND er.finished = 0
        """, (user_id,))
        data = cur.fetchall()
        cur.close()
        conn.close()

        # ===== 步骤3：生成任务列表（含原始内容） =====
        task_list = []
        for item in data:
            next_info = judge_next_review_time(item)
            task_entry = {
                "error_id": item["id"],
                "error_type": item["error_type"],
                "source_type": item.get("source_type"),
                "source_id": item.get("source_id"),
            }

            # 关联作文内容
            if item.get("source_type") == "writing" and item.get("writing_content"):
                task_entry["content_preview"] = (
                    item["writing_content"][:200] if item["writing_content"] else ""
                )
                task_entry["score"] = item.get("writing_score")
                task_entry["comment"] = item.get("writing_comment")
                task_entry["revise_content"] = item.get("writing_revise")

            # 关联翻译内容
            if item.get("source_type") == "trans" and item.get("trans_origin"):
                task_entry["content_preview"] = (
                    item["trans_origin"][:200] if item["trans_origin"] else ""
                )
                task_entry["user_trans"] = item.get("trans_user_trans")
                task_entry["score"] = item.get("trans_score")
                task_entry["suggest"] = item.get("trans_suggest")

            if next_info:
                task_entry["review_node"] = next_info

            task_list.append(task_entry)

        return {"tasks": task_list, "synced_count": synced_count}


def judge_next_review_time(record):
    """判断该错题是否到达下一轮复习时间节点（艾宾浩斯遗忘曲线）

    复习周期：创建后1天 → 3天 → 7天 → 15天
    只有当对应时间已过且该阶段尚未完成时，才返回待复习任务。
    """
    now = datetime.now()
    first_wrong = record.get("first_wrong")
    if not isinstance(first_wrong, datetime):
        return None

    if record["review_d1"] is None and now >= first_wrong + timedelta(days=1):
        return (record["id"], now, "第1轮(1天后)复盘")
    if record["review_d1"] is not None and record["review_d3"] is None and now >= first_wrong + timedelta(days=3):
        return (record["id"], now, "第2轮(3天后)复盘")
    if record["review_d3"] is not None and record["review_d7"] is None and now >= first_wrong + timedelta(days=7):
        return (record["id"], now, "第3轮(7天后)复盘")
    if record["review_d7"] is not None and record["review_d15"] is None and now >= first_wrong + timedelta(days=15):
        return (record["id"], now, "第4轮(15天后)复盘")
    return None


class KBAgent:
    @staticmethod
    def stat_hot_error():
        """统计全平台高频错题，生成知识库增补优化建议"""
        conn = get_db_connection()
        cur = conn.cursor(cursor=pymysql.cursors.DictCursor)
        cur.execute(
            """
        SELECT error_type,COUNT(*) cnt FROM error_record GROUP BY error_type ORDER BY cnt DESC
        """
        )
        res = cur.fetchall()
        cur.close()
        conn.close()

        suggest_text = "===== 知识库迭代增补建议 =====\n"
        type_map = {
            "vocab": "词汇类错题",
            "grammar": "语法类错题",
            "method": "解题思路方法类错题",
            "careless": "粗心失误类错题",
        }
        for row in res:
            et = row["error_type"]
            cnt = row["cnt"]
            suggest_text += f"{type_map.get(et, et)} 总量：{cnt} 条\n"
            if et == "vocab":
                suggest_text += "建议扩充高频生词、熟词僻义、固定搭配素材\n\n"
            elif et == "grammar":
                suggest_text += "建议补充对应语法例题、长难句拆解案例\n\n"
            elif et == "method":
                suggest_text += "建议补充对应题型解题步骤、干扰项识别技巧\n\n"
        if not res:
            suggest_text += "暂无错题数据，无需增补\n"
        return suggest_text
