import functools
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()
BASE_URL = os.getenv("RAGFLOW_BASE_URL")
API_KEY = os.getenv("RAGFLOW_API_KEY")
KB_DATASET_ID = os.getenv("KB_DATASET_ID")

# 超时配置：连接 5s，读取 30s（KB检索一般不会超过30s）
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 30
SEARCH_TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)


def _get_headers():
    """统一请求头：RAGFlow API 鉴权"""
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def check_ragflow_health() -> dict:
    """快速检测 RAGFlow 服务是否可达，避免长时间等待超时"""
    if not BASE_URL:
        return {"ok": False, "error": "RAGFLOW_BASE_URL 未配置"}
    try:
        # RAGFlow 文档服务健康检查端点
        resp = requests.get(
            f"{BASE_URL}/api/v1/version",
            headers=_get_headers(),
            timeout=(3, 5),
        )
        if resp.status_code == 200:
            return {"ok": True, "version": resp.json().get("data", {}).get("version", "unknown")}
        return {"ok": False, "error": f"RAGFlow 响应异常 HTTP {resp.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": f"无法连接 RAGFlow ({BASE_URL})，请确认 Docker 容器正在运行"}
    except requests.exceptions.Timeout:
        return {"ok": False, "error": f"连接 RAGFlow ({BASE_URL}) 超时，请检查服务状态"}
    except Exception as e:
        return {"ok": False, "error": f"健康检查异常: {str(e)}"}


@functools.lru_cache(maxsize=1)
def _cached_health_check() -> dict:
    """带缓存（60s内只检查一次）的健康检查"""
    return check_ragflow_health()


def search_kb(query: str, top_k: int = 3):
    """检索唯一知识库（所有业务共用同一个知识库），带快速失败和超时控制"""
    if not KB_DATASET_ID:
        return {"error": "KB_DATASET_ID 未在 .env 中配置"}
    if not API_KEY:
        return {"error": "RAGFLOW_API_KEY 未在 .env 中配置"}

    url = f"{BASE_URL}/api/v1/datasets/{KB_DATASET_ID}/search"
    payload = {
        "question": query,
        "top_k": top_k,
    }
    start = time.time()
    try:
        resp = requests.post(url, json=payload, headers=_get_headers(), timeout=SEARCH_TIMEOUT)
        elapsed = time.time() - start
        if resp.status_code == 200:
            result = resp.json()
            result["_elapsed"] = round(elapsed, 2)
            return result
        return {"error": f"RAGFlow检索失败 HTTP {resp.status_code}: {resp.text[:200]}", "_elapsed": round(elapsed, 2)}
    except requests.exceptions.ConnectionError:
        return {"error": f"无法连接RAGFlow服务 ({BASE_URL})，请确认Docker容器正在运行", "_elapsed": round(time.time() - start, 2)}
    except requests.exceptions.Timeout:
        return {"error": f"RAGFlow检索超时（{CONNECT_TIMEOUT}s连接 / {READ_TIMEOUT}s读取）", "_elapsed": round(time.time() - start, 2)}
    except Exception as e:
        return {"error": f"检索异常: {str(e)}", "_elapsed": round(time.time() - start, 2)}


# 以下函数全部指向同一个知识库，仅查询内容不同，保留接口兼容性

def search_writing_kb(query: str):
    """作文相关检索"""
    return search_kb(query)


def search_trans_kb(query: str):
    """翻译相关检索"""
    return search_kb(query)


def search_rule_kb(query: str):
    """评分规则检索"""
    return search_kb(query)


def search_grammar_kb(query: str):
    """语法相关检索"""
    return search_kb(query)


def search_vocab_kb(query: str):
    """词汇相关检索"""
    return search_kb(query)


def search_paper_kb(query: str):
    """真题试卷检索"""
    return search_kb(query)
