from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os
import pymysql
from agent_graph import multi_agent_graph
from db import get_db_connection, init_table
from ragflow_api import check_ragflow_health, search_kb, KB_DATASET_ID

load_dotenv()
app = FastAPI(title="四六级抗遗忘多智能体RAG复习系统")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_db_migrate():
    """应用启动时自动执行数据库迁移（新增列等兼容操作）"""
    try:
        init_table()
        print("[启动] 数据库表结构已检查/迁移完成")
    except Exception as e:
        print(f"[启动] 数据库迁移警告: {e}")


# ===================== 路由接口定义（带角色权限 + LangGraph多智能体编排） =====================
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/login")
async def login(username: str = Form(...), password: str = Form(...)):
    conn = get_db_connection()
    cur = conn.cursor(cursor=pymysql.cursors.DictCursor)
    cur.execute("SELECT * FROM user WHERE username=%s AND password=%s", (username, password))
    user = cur.fetchone()
    cur.close()
    conn.close()
    if not user:
        return HTMLResponse("账号密码错误 <br><a href='/'>返回首页</a>")
    # 根据角色跳转携带身份参数
    if user["role"] == "admin":
        return RedirectResponse("/?is_admin=1&uid="+str(user["id"]), status_code=302)
    else:
        return RedirectResponse("/?is_admin=0&uid="+str(user["id"]), status_code=302)

@app.post("/api/submit_writing")
async def submit_writing(user_id: int = Form(...), content: str = Form(...)):
    graph_state = multi_agent_graph.invoke({
        "task_type": "writing",
        "biz_type": "writing",
        "user_id": user_id,
        "content": content,
        "query": content,
    })
    res_data = graph_state["result"]
    score = res_data["score"]
    comment = res_data["comment"]
    revise = res_data["revise"]

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO writing_submit(user_id,content,score,comment,revise_content)
    VALUES(%s,%s,%s,%s,%s)
    """, (user_id, content, score, comment, revise))
    writing_id = cur.lastrowid

    # 得分<=9分（15分制不及格）自动创建错题记录，纳入艾宾浩斯复习计划
    auto_error_id = None
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        numeric_score = None
    if numeric_score is not None and numeric_score <= 9:
        cur.execute("""
        INSERT INTO error_record(user_id, error_type, source_type, source_id, first_wrong)
        VALUES(%s, 'method', 'writing', %s, NOW())
        """, (user_id, writing_id))
        auto_error_id = cur.lastrowid

    conn.commit()
    cur.close()
    conn.close()
    return {
        "code": 200, "score": score, "comment": comment, "revise": revise,
        "auto_error_id": auto_error_id,
    }

@app.post("/api/submit_trans")
async def submit_trans(
    user_id: int = Form(...),
    origin: str = Form(...),
    user_trans: str = Form(...)
):
    graph_state = multi_agent_graph.invoke({
        "task_type": "trans",
        "biz_type": "trans",
        "user_id": user_id,
        "origin": origin,
        "user_trans": user_trans,
        "query": origin,
    })
    res_data = graph_state["result"]
    score = res_data["score"]
    suggest = res_data["suggest"]
    better_trans = res_data["better_trans"]

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO trans_submit(user_id,origin_text,user_trans,score,suggest)
    VALUES(%s,%s,%s,%s,%s)
    """, (user_id, origin, user_trans, score, suggest))
    trans_id = cur.lastrowid

    # 得分<=9分（15分制不及格）自动创建错题记录
    auto_error_id = None
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        numeric_score = None
    if numeric_score is not None and numeric_score <= 9:
        cur.execute("""
        INSERT INTO error_record(user_id, error_type, source_type, source_id, first_wrong)
        VALUES(%s, 'grammar', 'trans', %s, NOW())
        """, (user_id, trans_id))
        auto_error_id = cur.lastrowid

    conn.commit()
    cur.close()
    conn.close()
    return {
        "code": 200, "score": score, "suggest": suggest, "better_trans": better_trans,
        "auto_error_id": auto_error_id,
    }

@app.get("/api/review_task/{user_id}")
async def get_review_task(user_id: int):
    graph_state = multi_agent_graph.invoke({"task_type": "review", "user_id": user_id})
    return graph_state["result"]


# ===================== 错题管理接口 =====================

@app.post("/api/error_record")
async def create_error_record(
    user_id: int = Form(...),
    error_type: str = Form(...),
    question_id: int = Form(None),
):
    """手动创建错题记录（vocab/grammar/method/careless）"""
    valid_types = ("vocab", "grammar", "method", "careless")
    if error_type not in valid_types:
        return JSONResponse(
            {"code": 400, "msg": f"error_type 必须为 {valid_types} 之一"},
            status_code=400,
        )

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO error_record(user_id, question_id, error_type, first_wrong) VALUES(%s,%s,%s,NOW())",
        (user_id, question_id, error_type),
    )
    new_id = cur.lastrowid
    conn.commit()
    cur.close()
    conn.close()
    return {"code": 200, "id": new_id, "msg": "错题记录已创建，已纳入艾宾浩斯复习计划"}


@app.get("/api/error_records/{user_id}")
async def list_error_records(user_id: int, finished: int = Query(None)):
    """查询用户的错题记录，可按完成状态筛选（0=未完成, 1=已完成），关联作文/翻译原始内容"""
    conn = get_db_connection()
    cur = conn.cursor(cursor=pymysql.cursors.DictCursor)

    base_query = """
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
    """

    if finished is not None:
        cur.execute(
            base_query + " WHERE er.user_id=%s AND er.finished=%s ORDER BY er.first_wrong DESC",
            (user_id, finished),
        )
    else:
        cur.execute(
            base_query + " WHERE er.user_id=%s ORDER BY er.first_wrong DESC",
            (user_id,),
        )
    records = cur.fetchall()

    # 格式化日期字段
    for r in records:
        for key in ("first_wrong", "review_d1", "review_d3", "review_d7", "review_d15"):
            if r.get(key):
                r[key] = str(r[key])

    cur.close()
    conn.close()

    # 统计各阶段待复习数量
    pending = {
        "total": len([r for r in records if not r["finished"]]),
        "d1": len([r for r in records if r["review_d1"] is None]),
        "d3": len([r for r in records if r["review_d1"] is not None and r["review_d3"] is None]),
        "d7": len([r for r in records if r["review_d3"] is not None and r["review_d7"] is None]),
        "d15": len([r for r in records if r["review_d7"] is not None and r["review_d15"] is None]),
    }

    return {"code": 200, "records": records, "pending_stats": pending}


@app.post("/api/review/complete")
async def mark_review_complete(
    error_id: int = Form(...),
    review_stage: str = Form(...),
):
    """标记某个复习节点已完成：d1 / d3 / d7 / d15

    完成d15后自动将finished置为1（艾宾浩斯4轮复习结束）
    """
    valid_stages = ("d1", "d3", "d7", "d15")
    if review_stage not in valid_stages:
        return JSONResponse(
            {"code": 400, "msg": f"review_stage 必须为 {valid_stages} 之一"},
            status_code=400,
        )

    conn = get_db_connection()
    cur = conn.cursor(cursor=pymysql.cursors.DictCursor)

    # 检查记录是否存在
    cur.execute("SELECT * FROM error_record WHERE id=%s", (error_id,))
    record = cur.fetchone()
    if not record:
        cur.close()
        conn.close()
        return JSONResponse({"code": 404, "msg": "错题记录不存在"}, status_code=404)

    # 更新对应阶段
    stage_column = f"review_{review_stage}"
    cur.execute(
        f"UPDATE error_record SET {stage_column}=NOW() WHERE id=%s",
        (error_id,),
    )

    # d15完成后标记finished
    if review_stage == "d15":
        cur.execute("UPDATE error_record SET finished=1 WHERE id=%s", (error_id,))

    conn.commit()

    # 查询更新后的状态
    cur.execute("SELECT * FROM error_record WHERE id=%s", (error_id,))
    updated = cur.fetchone()
    cur.close()
    conn.close()

    return {
        "code": 200,
        "msg": f"第{review_stage}轮复习已标记完成",
        "finished": bool(updated["finished"]),
        "next_stage": None if updated["finished"] else _get_next_stage(updated),
    }


def _get_next_stage(record) -> str | None:
    """获取下一个待复习阶段"""
    if record["review_d1"] is None:
        return "d1"
    if record["review_d3"] is None:
        return "d3"
    if record["review_d7"] is None:
        return "d7"
    if record["review_d15"] is None:
        return "d15"
    return None

# 管理员专属接口，增加权限校验
@app.get("/api/kb_suggest")
async def get_kb_suggest(is_admin: str = Query(...)):
    graph_state = multi_agent_graph.invoke({"task_type": "kb_admin", "is_admin": is_admin})
    return graph_state["result"]

@app.post("/api/rag_search")
async def rag_global_search(biz_type: str = Form(...), query: str = Form(...)):
    graph_state = multi_agent_graph.invoke({
        "task_type": "rag_search",
        "biz_type": biz_type,
        "query": query,
    })

    # 构建返回结果，包含错误追踪信息
    result = graph_state.get("result") or {}
    error = graph_state.get("error")
    trace = graph_state.get("trace", [])

    response = {
        "text": result.get("text", ""),
        "ai_answer": result.get("ai_answer", ""),
        "raw": result.get("raw"),
        "trace": trace,
    }

    if error:
        response["error"] = error

    return JSONResponse(response)

@app.post("/api/agent/invoke")
async def invoke_agent_graph(request: Request):
    payload = await request.json()
    graph_state = multi_agent_graph.invoke(payload)
    return {
        "result": graph_state.get("result"),
        "trace": graph_state.get("trace", []),
        "rag_context": graph_state.get("rag_context"),
        "error": graph_state.get("error"),
    }


@app.get("/api/health")
async def health_check():
    """系统健康检查 —— 诊断RAGFlow、数据库、知识库状态"""
    import requests as req

    status = {
        "ragflow": {"ok": False, "detail": ""},
        "database": {"ok": False, "detail": ""},
        "kb_dataset": {"ok": False, "detail": ""},
        "llm_chat": {"ok": False, "detail": ""},
    }

    # 1. RAGFlow 健康检查
    try:
        health = check_ragflow_health()
        status["ragflow"]["ok"] = health.get("ok", False)
        status["ragflow"]["detail"] = health.get("version") if health.get("ok") else health.get("error", "未知错误")
    except Exception as e:
        status["ragflow"]["detail"] = str(e)

    # 2. 数据库连接检查
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        status["database"]["ok"] = True
        status["database"]["detail"] = "连接正常"
    except Exception as e:
        status["database"]["detail"] = str(e)

    # 3. 知识库检查（测试检索）
    if status["ragflow"]["ok"] and KB_DATASET_ID:
        try:
            kb_result = search_kb("test", top_k=1)
            if "error" in kb_result:
                status["kb_dataset"]["detail"] = kb_result["error"]
            else:
                chunks = kb_result.get("data", {}).get("chunks", [])
                status["kb_dataset"]["ok"] = True
                status["kb_dataset"]["detail"] = f"知识库ID={KB_DATASET_ID}, 测试检索到 {len(chunks)} 条结果"
        except Exception as e:
            status["kb_dataset"]["detail"] = str(e)
    elif not KB_DATASET_ID:
        status["kb_dataset"]["detail"] = "KB_DATASET_ID 未配置"

    # 4. LLM Chat Assistant 检查（快速测试）
    base_url = os.getenv("RAGFLOW_BASE_URL")
    api_key = os.getenv("RAGFLOW_API_KEY")
    chat_id_llm = os.getenv("RAGFLOW_CHAT_ID_LLM", "").strip()
    if base_url and api_key and chat_id_llm:
        try:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            url = f"{base_url}/api/v1/chats_openai/{chat_id_llm}/chat/completions"
            payload = {
                "model": os.getenv("LLM_MODEL_NAME", "deepseek-v4-flash@shiki@DeepSeek"),
                "messages": [{"role": "user", "content": "回复OK"}],
                "stream": False,
            }
            resp = req.post(url, json=payload, headers=headers, timeout=(5, 15))
            if resp.status_code == 200:
                status["llm_chat"]["ok"] = True
                status["llm_chat"]["detail"] = f"Chat ID={chat_id_llm[:12]}..., 响应正常"
            else:
                status["llm_chat"]["detail"] = f"HTTP {resp.status_code}: {resp.text[:150]}"
        except req.exceptions.Timeout:
            status["llm_chat"]["detail"] = "LLM调用超时(15s)，可能模型响应过慢"
        except Exception as e:
            status["llm_chat"]["detail"] = str(e)
    else:
        missing = []
        if not chat_id_llm:
            missing.append("RAGFLOW_CHAT_ID_LLM")
        if not api_key:
            missing.append("RAGFLOW_API_KEY")
        status["llm_chat"]["detail"] = f"缺少配置: {', '.join(missing)}"

    all_ok = all(v["ok"] for v in status.values())
    return JSONResponse({
        "overall": "healthy" if all_ok else "degraded",
        "services": status,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
