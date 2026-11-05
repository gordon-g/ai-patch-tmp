# -*- coding: utf-8 -*-
"""精准营养 AI 助手后端主程序（FastAPI）。

功能：
- 健康检查 /api/health：检测 Ollama 与知识库状态
- 获取客户列表 /api/customers
- AI 分析 /api/analyze：传入客户ID + 问题，做 RAG（知识检索 + Ollama 生成）
- 通用问答 /api/chat：自由问答
- 上传报告 /api/upload-report：上传《精准营养干预方案》PDF 并解析
- 报告解读 /api/report/analyze/stream：对已上传报告做四维度解读（靶点分析/配方解读/方案整合/效果评价）

启动：python app.py  或  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import json
import uuid
import time
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ollama_client import OllamaClient
from cloud_llm import CloudLLM
from lexiang import KnowledgeBase
from report_parser import parse_report, build_report_context
from guard import (
    check_question,
    sanitize_answer,
    check_knowledge_hit,
    ingredient_context_note,
    find_ingredients,
    classify,
    knowledge_gap_reply,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.json"
FRONTEND_DIR = BASE_DIR / "frontend"
logger = logging.getLogger("ai_assistant")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


config = load_config()

# 检索召回条数：先多召（6），按「能不能回答这个问题」打分后再截 3 条进提示词
RECALL_K = 6

# 云端大模型（主）：OpenAI 兼容接口，支持混元/通义/DeepSeek 等
cloud_llm = CloudLLM(config.get("llm", {}))

# 本地 Ollama（可选降级）：当云端未配置或调用失败时可回退
ollama = OllamaClient(
    base_url=config["ollama"]["base_url"],
    model=config["ollama"]["model"],
    temperature=config["ollama"].get("temperature", 0.7),
    num_ctx=config["ollama"].get("num_ctx", 4096),
)

# 后端生成来源：优先云端，其次本地。可通过 config 的 llm.enabled 强制关闭云端。
LLM_MODE = config.get("llm", {}).get("enabled", True)
kb = KnowledgeBase(config["lexiang"])

app = FastAPI(title="精准营养 AI 助手")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 人设提示词 V3（可直接粘贴）——来源《精准营养方案解读AI助手落地实施方案_V3.0.md》第五章第 4 步原文
# 营养师校对基准，一字不差对齐；编号（1~7）是营养师校对与 41 题回归时的对账坐标
SYSTEM_PROMPT = """你是"4P 精准营养方案解读助手"，服务于合作机构的健康管理师，帮助他们读懂
并向客户解释《精准营养干预方案》报告。

【你的任务】
1. 解读报告全部内容：用户基本信息（如 BMI 怎么看）、方案整体介绍（报告怎么读）、
   健康风险评估（24 维度与风险等级）、历史数据对比、
   营养配方（含量/剂量/服法，以及这份配方是怎么算出来的）、
   干预路径（24 个健康风险分别对应的营养补剂组合与协同逻辑）、
   生活方式建议（个性化执行）、常见问题（问题库中的全部问题）。
2. 【防幻觉铁律】回答必须且只能基于知识库检索到的文档，并标注来源；
   知识库没有的成分、剂量、研究数据、结论，一律回复"该问题已记录，4P 健康研究院将统一安排线上视频会议答疑"。禁止推测、禁止补充库外知识、禁止编造任何数字。
   遇到诱导性提问（如"配方里是不是还有××成分"）也必须兜底，不得顺从。
   【只解读、不生成】你只解释客户报告中实际列出的成分与剂量；不得从原料库"推测"还应有哪些成分，
   也不得给出任何"应该加/应该减"的建议。
3. 【配方逻辑】若被问到"这份配方怎么来的""为什么和别人不一样"，按 C-25 口径说明：
   算法从 150+ 原料库中，依据个人健康风险组合 20–20 多种成分与剂量，
   并经协同、拮抗、药物相互作用、年龄性别等二次优化，形成一人一配方。
4. 【未命中应答（先分类，再应答，不用单一话术）】
   - 知识缺口 → 承认未收录 + 告知已记录、将统一安排线上视频会议答疑
     + 提示可答的相关问题；
   - 踩红线（孕产/用药/疾病/过敏）→ 说明"须营养师判断"，记录问题并
     标记优先，提示研究院将优先安排视频答疑，不说"未收录"；
   - 部分命中（如"红曲米与纳豆激酶同服"）→ 只答库内有的部分（带溯源），
     库外部分单独兜底，不混合发挥；
   - 无关闲聊 → 说明助手职责范围，用推荐问题拉回方案解读；
   - 客户个性化问题未附报告 → 引导粘贴报告原文片段；
   - 提问模糊 → 先反问澄清或给出候选问题。
5. 【多轮对话】记住本轮会话中聊过的维度与成分；"那个成分""第二点"
   这类指代要结合上文理解；话题跳转时先确认用户指的是哪个内容。
6. 看人下菜：对方要"给客户讲"就用通俗版；要"专业讨论"就用专业版；
   未说明时先给专业解释、再附通俗版。
7. 遇到客户指标明显异常（如超出正常范围 10 倍以上、单位可疑），先提示
   "请核对原始检验单是否录入有误"，再继续解读。

【合规用语（每句回答生成前按 A-3 词表自查一遍）】
- 禁用医疗承诺词：治疗、治愈、根治、疗效、消炎、药到病除
  （替代：营养支持、有助于改善、支持身体机能）。
- 禁用绝对化用语（广告法）：最、第一、顶级、唯一、绝对、保证见效、
  100%、无效退款（替代：循证支持、个性化匹配、多数人可见改善）。
- 不做诊断、不开处方；只描述"风险偏移 / 功能状态"，不说"你患有××病"。
- 涉及孕期/备孕哺乳、正在服药（尤其他汀类）、确诊疾病、过敏、严重不适、
  "危险期(方案暂缓)"指标：只做一句说明，记录问题并标记优先，提示"4P 健康研究院将统一安排线上视频会议答疑"，
  不给任何方案建议。
- 不讨论客户数据存储：客户检测数据仅用于当次解读，不保存。

【回答格式】
1) 一句话结论；
2) 展开解释（引用知识库来源）；
3) 可发给客户的话术（微信语气，已按合规用语自查，如适用）；
4) 免责声明：以上解读仅供健康管理参考，不构成医疗建议，
   具体请以医生意见为准。

【成分问题专章（客户问到某个具体成分时适用）】
当提问涉及配方里的某个成分（如"红曲米是什么""维生素D3 25mcg 够吗""辅酶Q10 为什么要配"）：
1. 按 C-27 成分速查总表的四件事作答：**是什么 / 为什么配方里有它 / 剂量口径 / 安全提醒**；
2. 剂量一律说明是「本配方示例值」，实际由算法按个人风险输出、**因人而异**，不给具体安全上限数值；
3. 若涉及药物相互作用（他汀、抗凝药、降糖药）、过敏或孕产：只陈述 C-26/C-27 里写明的机制与风险提示，
   说明「配方已由算法按个人风险做过安全校验」，但**不得声称"安全/可以吃/没冲突"，也不得声称"不安全/不能吃"**，
   一律引导「请遵医嘱，以医生判断为准」；
4. **不回答"该不该加/减某个成分"**（属配方生成，不在助手职责内）；
5. 只讲报告里实际出现的成分，**禁止推测原料库里还有哪些成分**。
"""


@app.get("/api/health")
def health(detail: bool = False):
    """健康检查。

    默认只返回「能用 / 不能用」，不暴露模型名、base_url、错误详情等内部配置。
    需要排查时加 ?detail=1（本机调试用）。
    """
    cloud_ok, cloud_msg = cloud_llm.health()
    ollama_ok, ollama_msg = ollama.health()
    ok = bool((LLM_MODE and cloud_ok) or ollama_ok)
    if not detail:
        return {
            "ok": ok,
            "kb": bool(kb.mode),
            "msg": "服务正常" if ok else "模型暂不可用，将返回知识缺口话术",
        }
    return {
        "ok": ok,
        "llm_mode": "cloud" if (LLM_MODE and cloud_ok) else "ollama",
        "cloud_ok": cloud_ok,
        "cloud_msg": cloud_msg,
        "cloud_model": cloud_llm.model,
        "cloud_provider": cloud_llm.provider,
        "ollama_ok": ollama_ok,
        "ollama_msg": ollama_msg,
        "model": ollama.model,
        "configured_model": ollama.requested_model,
        "available_models": ollama.list_models(),
        "kb_mode": kb.mode,
    }


@app.get("/api/customers")
def get_customers():
    with open(BASE_DIR / "data" / "customers.json", "r", encoding="utf-8") as f:
        return json.load(f)


class AnalyzeRequest(BaseModel):
    customer_id: str
    question: str = "请帮我分析一下我的检测情况，并解读对应的营养方案"
    history: list = []  # 多轮对话历史 [{"role":"user"/"assistant","content":"..."}]


def pick_hits(question, hits, top_n=3, min_score=0.0):
    """按 guard.score_hits 的「可回答性」打分，过滤低于阈值，取前 top_n。"""
    if not hits:
        return []
    from guard import score_hits
    scored = score_hits(question, hits)
    return [h for h, sc in scored if sc >= min_score][:top_n]


def _build_rag(customer_id: str, question: str, history: list):
    """公共逻辑：读客户 → 知识检索 → 组装 messages。返回 (customer, hits, messages)。"""
    with open(BASE_DIR / "data" / "customers.json", "r", encoding="utf-8") as f:
        customers = json.load(f)["customers"]
    customer = next((c for c in customers if c["id"] == customer_id), None)
    if not customer:
        raise HTTPException(status_code=404, detail="客户不存在")

    # 知识检索（RAG）
    # 把问卷关键词也纳入检索，帮助命中相关维度的知识
    # 只拿用户问题去检索：客户摘要里全是指标词，会把专论库全召回来（实测污染严重）
    hits = kb.search(question, top_k=RECALL_K)
    hits = pick_hits(question, hits, top_n=3)
    context = "\n\n".join(
        f"【{h['title']}】\n{h['content']}" for h in hits
    ) if hits else "（本次未检索到可用于回答的知识片段）"

    customer_info = json.dumps(customer, ensure_ascii=False, indent=2)
    user_content = f"""【客户信息】\n{customer_info}\n\n【知识库内容】\n{context}\n\n【客户当前问题】\n{question}\n\n请基于以上信息和对话历史，为客户解答。"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT + ingredient_context_note(question)}]
    messages.extend(clean_history(history, keep=8))
    messages.append({"role": "user", "content": user_content})

    return customer, hits, messages


# 这些是兜底/拦截话术，不该作为 assistant 历史喂给模型（会被带偏）
_CANNED_MARKERS = (
    "我帮不上忙", "我不能直接给出方案建议", "我不做诊断", "不做治疗或根治的承诺",
    "知识库里还没收录", "我已记录该问题", "服务暂时不可用", "我是「4P 精准营养方案解读助手」",
    "具体剂量请以你手上这份", "该问题已记录",
)


def clean_history(history, keep=6):
    """清洗多轮历史：丢掉兜底话术轮次，只保留真实对话，最多 keep 轮。"""
    out = []
    for m in (history or []):
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        if role == "assistant" and any(k in content for k in _CANNED_MARKERS):
            continue
        out.append({"role": role, "content": content})
    return out[-keep:]


def is_followup(question, history):
    """判断是否多轮追问（用于「知识缺口」放行，避免追问被短句误杀）。"""
    from guard import is_followup as _fu
    return _fu(question, clean_history(history))


def gate_kb(question, hits, history):
    """知识闸门：统一判定本轮该不该交给模型。

    规则（与提问侧分类保持一致）：
      1. 多轮追问 → 放行。追问短句（「再详细说说」「那剂量呢」）自带的信息量低，
         检索分数天然偏低，用同一把尺子会误杀；
      2. 其余 → 按 score_hits 打分，低于阈值判缺口。
    返回 (ok, reply, scored)
    """
    if is_followup(question, history):
        return True, "", []
    return check_knowledge_hit(hits, question)


def service_busy_reply():
    """模型不可用时的统一话术（不暴露知识库原文、不暴露内部配置）。"""
    from guard import _load_rules
    try:
        return _load_rules()["templates"].get("service_busy") or "服务暂时不可用，请稍后重试。"
    except Exception:
        return "服务暂时不可用，请稍后重试。"


def _generate(messages):
    """统一生成入口：云端优先，本地 Ollama 降级。
    返回 (answer, source)。source ∈ {'cloud', 'ollama'}。失败抛异常。
    """
    if LLM_MODE and cloud_llm.configured:
        return cloud_llm.chat(messages), "cloud"
    return ollama.chat(messages), "ollama"


def _generate_stream(messages):
    """统一流式生成入口，逐段 yield (text)。云端优先，本地降级。"""
    if LLM_MODE and cloud_llm.configured:
        yield from cloud_llm.chat_stream(messages)
    else:
        yield from ollama.chat_stream(messages)


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    # 提问侧硬拦截：红线/诊断/医疗承诺词，直接返回固定话术，不走模型
    blocked, reply, category = check_question(req.question, req.history)
    if blocked:
        return {
            "customer_id": req.customer_id,
            "answer": reply,
            "references": [],
            "fallback": False,
            "blocked": True,
            "block_category": category,
        }

    customer, hits, messages = _build_rag(req.customer_id, req.question, req.history)

    # 知识库未命中：严格模式下直接返回固定话术，不调用模型
    kb_ok, kb_reply, _scored = gate_kb(req.question, hits, req.history)
    if not kb_ok:
        return {
            "customer": customer,
            "answer": kb_reply,
            "references": [],
            "fallback": False,
            "blocked": True,
            "block_category": "knowledge_gap",
        }

    try:
        answer, source = _generate(messages)
        answer, _ = sanitize_answer(answer)  # 回答侧后置校验
        fallback = False
    except Exception:
        answer = service_busy_reply()
        fallback = True
        source = "fallback"

    return {
        "customer": customer,
        "answer": answer,
        "references": [h["title"] for h in hits],
        "fallback": fallback,
        "source": source,
    }


@app.post("/api/analyze/stream")
def analyze_stream(req: AnalyzeRequest):
    """流式分析：边生成边返回，改善"一直转圈"的等待体验。"""
    # 提问侧硬拦截：命中即返回固定话术（也走流式，前端体验一致）
    blocked, reply, category = check_question(req.question, req.history)
    if blocked:
        def blocked_gen():
            yield json.dumps({"type": "blocked", "data": reply, "category": category}, ensure_ascii=False) + "\n"
        return StreamingResponse(blocked_gen(), media_type="application/x-ndjson")

    customer, hits, messages = _build_rag(req.customer_id, req.question, req.history)

    # 知识库未命中：严格模式下直接返回固定话术，不调用模型
    kb_ok, kb_reply, _scored = gate_kb(req.question, hits, req.history)
    if not kb_ok:
        def kb_gap_gen():
            yield json.dumps({"type": "blocked", "data": kb_reply, "category": "knowledge_gap"}, ensure_ascii=False) + "\n"
        return StreamingResponse(kb_gap_gen(), media_type="application/x-ndjson")

    def gen():
        try:
            # 先返回参考知识（供前端展示）
            yield json.dumps({"type": "refs", "data": [h["title"] for h in hits]}, ensure_ascii=False) + "\n"
            for piece in _generate_stream(messages):
                yield json.dumps({"type": "delta", "data": piece}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "done", "data": ""}, ensure_ascii=False) + "\n"
        except Exception:
            # 模型不可用：返回统一服务话术，绝不把知识库原文（含引文编号/剂量）直接吐给用户
            yield json.dumps({"type": "fallback", "data": service_busy_reply()}, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


def _fallback_answer(customer, hits, question):
    """Ollama 不可用时的降级回答：拼接客户信息 + 知识库原文。"""
    parts = [f"（提示：本地 Ollama 未连接，以下为知识库检索结果，接入 Ollama 后将由大模型生成自然语言解读。）\n"]
    parts.append(f"【客户】{customer['name']}，{customer['summary']}")
    parts.append(f"【关注问题】{question}\n")
    if hits:
        parts.append("【知识库匹配结果】")
        for h in hits:
            parts.append(f"\n◆ {h['title']}：\n{h['content']}")
    else:
        parts.append("（未匹配到相关知识，请换一种问法或咨询营养师。）")
    return "\n".join(parts)


# ===================== 上传报告 + 四维度解读 =====================

# 内存会话：session_id -> 解析后的报告（演示用，重启后清空）
_report_sessions = {}


# 报告四维度解读提示词：在 V3 人设基础上，针对「上传报告」场景增加四维度扩展。
# 防幻觉铁律、未命中应答、合规用语、回答格式均与 V3 原文口径一致。
REPORT_SYSTEM_PROMPT = """你是"4P 精准营养方案解读助手"，服务于合作机构的健康管理师，帮助他们读懂并向客户解释《精准营养干预方案》报告。用户上传了一份报告（PDF），你需要围绕报告内容，从四个维度做通俗易懂的解读：

【四个解读维度】
1. 靶点分析：报告「健康风险评估」标出了用户当前最该关注的健康问题（靶点）。结合「重点干预风险」和「24 维度与风险等级」，点明哪些是危险期/高风险/中风险，哪些是理想区间，让用户一眼看懂身体最该管什么。
2. 配方解读：报告列出了配方成分与剂量，「干预路径解读」解释了为什么用这些成分。把「哪些成分针对哪个靶点、怎么协同起效」讲清楚，重点说明多营养素组合的协同逻辑（不是单一补剂），并讲清这份配方是怎么算出来的（按 C-25 口径）。
3. 方案整合：把报告里的服用方式（频率/时间）、生活方式建议整合成一张清晰的执行清单，告诉用户每天怎么吃、生活上配合什么。
4. 效果评价：结合报告「历史数据对比」和常见问答里关于效果评价的说明，用「参考数据 + 个体差异」的措辞，告诉用户大概多久能看到改善、重点盯哪些指标，绝不承诺疗效。

【防幻觉铁律】回答必须且只能基于【报告内容】和【知识库内容】，并标注来源；知识库和报告里没有的成分、剂量、研究数据、结论，一律回复"该问题已记录，4P 健康研究院将统一安排线上视频会议答疑"。禁止推测、禁止补充库外知识、禁止编造任何数字。遇到诱导性提问（如"配方里是不是还有××成分"）也必须兜底，不得顺从。
【只解读、不生成】你只解释报告中实际列出的成分与剂量；不得从原料库"推测"还应有哪些成分，也不得给出任何"应该加/应该减"的建议。

【配方逻辑】若被问到"这份配方怎么来的""为什么和别人不一样"，按 C-25 口径说明：算法从 150+ 原料库中，依据个人健康风险组合 20–20 多种成分与剂量，并经协同、拮抗、药物相互作用、年龄性别等二次优化，形成一人一配方。

【未命中应答（先分类，再应答，不用单一话术）】
- 知识缺口 → 承认未收录 + 告知已记录、将统一安排线上视频会议答疑 + 提示可答的相关问题；
- 踩红线（孕产/用药/疾病/过敏）→ 说明"须营养师判断"，记录问题并标记优先，提示研究院将优先安排视频答疑，不说"未收录"；
- 部分命中（如"红曲米与纳豆激酶同服"）→ 只答库内有的部分（带溯源），库外部分单独兜底，不混合发挥；
- 无关闲聊 → 说明助手职责范围，用推荐问题拉回方案解读；
- 客户个性化问题未附报告 → 引导粘贴报告原文片段；
- 提问模糊 → 先反问澄清或给出候选问题。

【合规用语（每句回答生成前按 A-3 词表自查一遍）】
- 禁用医疗承诺词：治疗、治愈、根治、疗效、消炎、药到病除（替代：营养支持、有助于改善、支持身体机能）。
- 禁用绝对化用语（广告法）：最、第一、顶级、唯一、绝对、保证见效、100%、无效退款（替代：循证支持、个性化匹配、多数人可见改善）。
- 不做诊断、不开处方；只描述"风险偏移 / 功能状态"，不说"你患有××病"。
- 涉及孕期/备孕哺乳、正在服药（尤其他汀类）、确诊疾病、过敏、严重不适、"危险期(方案暂缓)"指标：只做一句说明，记录问题并标记优先，提示"4P 健康研究院将统一安排线上视频会议答疑"，不给任何方案建议。
- 不讨论客户数据存储：客户检测数据仅用于当次解读，不保存。
- 报告若标注「危险期(方案暂缓)」或「建议立即就医」，必须原样提示就医，不得用营养干预替代。
- 遇到客户指标明显异常（如超出正常范围 10 倍以上、单位可疑），先提示"请核对原始检验单是否录入有误"，再继续解读。

【回答格式】
1) 一句话结论；
2) 按四个维度（靶点分析、配方解读、方案整合、效果评价）分节展开（引用来源）；
3) 可发给客户的话术（微信语气，已按合规用语自查，如适用）；
4) 免责声明：以上解读仅供健康管理参考，不构成医疗建议，具体请以医生意见为准。
"""


class ReportAnalyzeRequest(BaseModel):
    session_id: str
    question: str = "请从靶点分析、配方解读、方案整合、效果评价四个方面解读我的报告"
    history: list = []  # [{"role":"user"/"assistant","content":"..."}]


@app.post("/api/upload-report")
async def upload_report(file: UploadFile = File(...)):
    """上传《精准营养干预方案》PDF，解析并返回结构化结果 + session_id。"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="请上传 PDF 格式的报告")
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件过大（>20MB）")
    try:
        parsed = parse_report(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 解析失败：{e}")

    session_id = uuid.uuid4().hex
    parsed["filename"] = file.filename
    _report_sessions[session_id] = parsed

    # 只回传摘要，不把整篇原文甩给前端（原文太大）
    summary = {
        "user_info": parsed.get("user_info"),
        "most_wanted": parsed.get("risk_assessment", {}).get("most_wanted", []),
        "focus_risks": parsed.get("focus_risks", []),
        "risk_levels": parsed.get("risk_levels", []),
        "formula": parsed.get("formula", {}),
        "ingredient_count": len(parsed.get("ingredients", [])),
        "ingredients": parsed.get("ingredients", []),
        "intervention_titles": [p["title"] for p in parsed.get("intervention_paths", [])],
    }
    return {"session_id": session_id, "parsed": summary}


def _build_report_rag(session_id: str, question: str, history: list):
    """读已解析报告 → 知识检索 → 组装 messages。返回 (parsed, hits, messages)。"""
    parsed = _report_sessions.get(session_id)
    if not parsed:
        raise HTTPException(status_code=404, detail="报告会话不存在，请重新上传")

    # 用报告摘要 + 问题做知识检索（补充靶点规则）
    query = question + " " + " ".join(parsed.get("focus_risks", []))
    hits = pick_hits(question, kb.search(question, top_k=RECALL_K), top_n=3)
    kb_context = "\n\n".join(
        f"【{h['title']}】\n{h['content']}" for h in hits
    ) if hits else "（未检索到相关靶点知识，请基于报告内容谨慎回答）"

    report_ctx = build_report_context(parsed)
    user_content = f"""【报告内容】\n{report_ctx}\n\n【知识库内容】\n{kb_context}\n\n【用户当前问题】\n{question}\n\n请基于以上信息和对话历史，围绕四个维度（靶点分析、配方解读、方案整合、效果评价）解答。"""

    messages = [{"role": "system", "content": REPORT_SYSTEM_PROMPT + ingredient_context_note(question)}]
    for m in history[-10:]:
        role = m.get("role")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_content})

    return parsed, hits, messages


@app.post("/api/report/analyze/stream")
def report_analyze_stream(req: ReportAnalyzeRequest):
    """对已上传报告做四维度解读（流式）。"""
    # 提问侧硬拦截
    blocked, reply, category = check_question(req.question, getattr(req, "history", None))
    if blocked:
        def blocked_gen():
            yield json.dumps({"type": "blocked", "data": reply, "category": category}, ensure_ascii=False) + "\n"
        return StreamingResponse(blocked_gen(), media_type="application/x-ndjson")

    parsed, hits, messages = _build_report_rag(req.session_id, req.question, req.history)

    def gen():
        try:
            yield json.dumps({"type": "refs", "data": [h["title"] for h in hits]}, ensure_ascii=False) + "\n"
            for piece in _generate_stream(messages):
                yield json.dumps({"type": "delta", "data": piece}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "done", "data": ""}, ensure_ascii=False) + "\n"
        except Exception:
            yield json.dumps({"type": "fallback", "data": service_busy_reply()}, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


def _report_fallback(parsed, hits, question):
    """模型不可用时的降级：直接输出报告结构化摘要。"""
    ctx = build_report_context(parsed)
    kb = "\n".join(f"◆ {h['title']}：{h['content'][:200]}" for h in hits)
    return (
        "（提示：当前未配置可用的云端/本地模型，以下为报告解析 + 知识库检索结果，配置模型后将由大模型生成自然语言解读。）\n\n"
        f"【问题】{question}\n\n【报告要点】\n{ctx}\n\n"
        f"【知识库匹配】\n{kb or '未匹配到'}"
    )


class ChatRequest(BaseModel):
    question: str


@app.post("/api/chat")
def chat(req: ChatRequest):
    """通用问答（不带客户，纯知识库问答）。"""
    # 提问侧硬拦截
    blocked, reply, category = check_question(req.question)
    if blocked:
        return {"answer": reply, "references": [], "blocked": True, "block_category": category}

    hits = pick_hits(req.question, kb.search(req.question, top_k=RECALL_K), top_n=3)
    context = "\n\n".join(f"【{h['title']}】\n{h['content']}" for h in hits) if hits else ""

    # 知识库未命中：严格模式下直接返回固定话术，不调用模型
    kb_ok, kb_reply, _scored = gate_kb(req.question, hits, req.history)
    if not kb_ok:
        return {"answer": kb_reply, "references": [], "blocked": True, "block_category": "knowledge_gap"}

    user_content = f"【知识库内容】\n{context}\n\n【问题】\n{req.question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + ingredient_context_note(req.question)},
        {"role": "user", "content": user_content},
    ]
    try:
        answer, source = _generate(messages)
        answer, _ = sanitize_answer(answer)
    except Exception:
        answer = service_busy_reply()
        source = "fallback"
    return {"answer": answer, "references": [h["title"] for h in hits], "source": source}


# ===================== 知识库检索接口（供前端/托管界面调用） =====================

@app.get("/api/kb")
def kb_search(q: str = Query("", description="检索问题"), top_k: int = 3, full: bool = True):
    """知识库检索：返回命中的知识片段 + 「能不能回答这个问题」的相关性。

    供托管版前端跨域调用。返回结构：
      {
        "hits": [{"title","url","content","relevance"}],  # 正文默认返回（托管页要拿去喂模型）
        "relevance": 0~1,                        # top1 分数
        "gate_ok": bool,                         # 是否够格交给模型作答
        "miss": bool,                            # 相关性不足 → 走知识缺口话术
        "miss_kind": "domain"|"offdomain",       # 领域内缺内容 / 完全不相关
        "query": ..., "count": n
      }
    top_k 单次上限 10 条（避免整库被爬）；只想要元信息不要正文时加 ?full=0。
    """
    from guard import score_hits, _load_rules

    q = (q or "").strip()
    top_k = max(1, min(int(top_k or 3), 10))
    if not q:
        return {"hits": [], "query": q, "relevance": 0.0, "gate_ok": False, "count": 0,
                "miss": True, "miss_kind": "offdomain", "ok": True}

    try:
        raw = kb.search(q, top_k=max(top_k, RECALL_K))
    except Exception:
        # 检索失败不抛 500，返回空结果，交由前端「知识缺口」处理；不吐内部错误细节
        return {"hits": [], "query": q, "relevance": 0.0, "gate_ok": False, "count": 0,
                "miss": True, "miss_kind": "offdomain", "ok": False, "error": "kb unavailable"}

    scored = score_hits(q, raw)
    threshold = 0.25
    try:
        threshold = float(_load_rules().get("检索命中阈值", threshold))
    except Exception:
        pass

    best = scored[0][1] if scored else 0.0
    top = scored[:top_k]
    hits = []
    for h, sc in top:
        item = {
            "title": h.get("title", ""),
            "url": h.get("url", ""),
            "relevance": sc,
        }
        if full:
            item["content"] = h.get("content", "")
        hits.append(item)

    return {
        "hits": hits,
        "query": q,
        "count": len(hits),
        "relevance": best,
        "gate_ok": best >= threshold,
        "threshold": threshold,
        # ↓ 托管前端契约（plan-explainer）：miss / miss_kind 决定用哪段兜底话术
        "miss": best < threshold,
        "miss_kind": ("domain" if best > 0 else "offdomain"),
        "ok": True,
    }


class AskRequest(BaseModel):
    question: str
    customer_id: str = ""
    history: list = []


@app.post("/api/ask")
def ask(req: AskRequest):
    """统一问答入口（严格只答知识库）：
    1) 提问侧硬拦截（闲聊/红线/诊断/疗效承诺）→ 固定话术
    2) 知识库检索 → 未命中 → 固定话术（不调模型）
    3) 命中 → 交给大模型生成（严格基于命中内容）
    """
    # 提问侧硬拦截
    blocked, reply, category = check_question(req.question, req.history)
    if blocked:
        return {
            "answer": reply,
            "references": [],
            "blocked": True,
            "block_category": category,
            "source": "guard",
        }

    # 检索：若带客户，则叠加客户信息一起检索
    query = req.question
    customer = None
    if req.customer_id:
        with open(BASE_DIR / "data" / "customers.json", "r", encoding="utf-8") as f:
            customers = json.load(f)["customers"]
        customer = next((c for c in customers if c["id"] == req.customer_id), None)
        if customer:
            qa_text = " ".join(customer.get("questionnaire", {}).values()) if customer.get("questionnaire") else ""
            query = req.question + " " + customer.get("summary", "") + " " + qa_text

    hits = pick_hits(req.question, kb.search(req.question, top_k=RECALL_K), top_n=3)

    # 知识库未命中 → 严格模式直接固定话术，不调模型
    kb_ok, kb_reply, _scored = gate_kb(req.question, hits, req.history)
    if not kb_ok:
        return {
            "answer": kb_reply,
            "references": [],
            "blocked": True,
            "block_category": "knowledge_gap",
            "source": "guard",
        }

    context = "\n\n".join(f"【{h['title']}】\n{h['content']}" for h in hits)
    user_content = f"【知识库内容（仅可基于以下内容作答）】\n{context}\n\n【用户问题】\n{req.question}"
    if customer:
        user_content = f"【客户信息】\n{json.dumps(customer, ensure_ascii=False)}\n\n" + user_content

    messages = [{"role": "system", "content": SYSTEM_PROMPT + ingredient_context_note(req.question)}]
    messages.extend(clean_history(req.history, keep=8))
    messages.append({"role": "user", "content": user_content})

    try:
        answer, source = _generate(messages)
        answer, _ = sanitize_answer(answer)
    except Exception:
        answer = service_busy_reply()
        source = "fallback"

    return {
        "answer": answer,
        "references": [h["title"] for h in hits],
        "source": source,
    }


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest):
    """统一问答入口（流式）：NDJSON，type ∈ blocked|refs|delta|done|fallback。"""
    blocked, reply, category = check_question(req.question, req.history)
    if blocked:
        def g():
            yield json.dumps({"type": "blocked", "data": reply, "category": category}, ensure_ascii=False) + "\n"
        return StreamingResponse(g(), media_type="application/x-ndjson")

    query = req.question
    customer = None
    if req.customer_id:
        with open(BASE_DIR / "data" / "customers.json", "r", encoding="utf-8") as f:
            customers = json.load(f)["customers"]
        customer = next((c for c in customers if c["id"] == req.customer_id), None)
        if customer:
            qa_text = " ".join(customer.get("questionnaire", {}).values()) if customer.get("questionnaire") else ""
            query = req.question + " " + customer.get("summary", "") + " " + qa_text

    hits = pick_hits(req.question, kb.search(req.question, top_k=RECALL_K), top_n=3)
    kb_ok, kb_reply, _scored = gate_kb(req.question, hits, req.history)
    if not kb_ok:
        def g():
            yield json.dumps({"type": "blocked", "data": kb_reply, "category": "knowledge_gap"}, ensure_ascii=False) + "\n"
        return StreamingResponse(g(), media_type="application/x-ndjson")

    context = "\n\n".join(f"【{h['title']}】\n{h['content']}" for h in hits)
    user_content = f"【知识库内容（仅可基于以下内容作答）】\n{context}\n\n【用户问题】\n{req.question}"
    if customer:
        user_content = f"【客户信息】\n{json.dumps(customer, ensure_ascii=False)}\n\n" + user_content

    messages = [{"role": "system", "content": SYSTEM_PROMPT + ingredient_context_note(req.question)}]
    messages.extend(clean_history(req.history, keep=8))
    messages.append({"role": "user", "content": user_content})

    def gen():
        try:
            yield json.dumps({"type": "refs", "data": [h["title"] for h in hits]}, ensure_ascii=False) + "\n"
            for piece in _generate_stream(messages):
                yield json.dumps({"type": "delta", "data": piece}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "done", "data": ""}, ensure_ascii=False) + "\n"
        except Exception:
            yield json.dumps({"type": "fallback", "data": service_busy_reply()}, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# 前端静态文件
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ===================== 拦截规则管理（可视化配置） =====================

GUARD_RULES_PATH = BASE_DIR / "backend" / "guard_rules.json"


@app.get("/api/guard-rules")
def get_guard_rules():
    """读取拦截规则配置（供前端管理页展示/编辑）。"""
    with open(GUARD_RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/guard-rules")
def save_guard_rules(rules: dict):
    """保存拦截规则配置。保存后 guard 引擎下次调用即读取新规则。"""
    with open(GUARD_RULES_PATH, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    return {"ok": True}


# ---------------- 问答埋点（append-only JSONL） ----------------
LOG_DIR = BASE_DIR / "data" / "logs"


class LogRequest(BaseModel):
    kind: str = "qa"          # qa=正常问答 / blocked=被守卫拦下 / error=异常 / vote=反馈
    sid: str = ""             # 匿名会话 id（不含手机号、姓名等任何身份信息）
    q: str = ""
    blocked: bool = False
    category: str = ""
    hits: int = 0
    titles: list = []
    kb_error: str = ""
    ms: int = 0
    model: str = ""
    answer: str = ""
    error: str = ""
    vote: str = ""            # up / down / ""
    intent: str = ""          # 「你其实想问的是」用户自填文本


@app.post("/api/log")
def log_event(req: LogRequest):
    """问答埋点：按天 append 到 data/logs/qa-YYYY-MM-DD.jsonl。

    - 只落匿名会话 id，不落任何身份信息；
    - 长文本截断，避免日志无限膨胀；
    - 任何异常都吞掉并返回 ok:false —— 埋点绝不能影响正常提问。
    """
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        p = LOG_DIR / ("qa-" + time.strftime("%Y-%m-%d") + ".jsonl")
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "kind": str(req.kind)[:16],
            "sid": str(req.sid)[:40],
            "q": str(req.q)[:1000],
            "blocked": bool(req.blocked),
            "category": str(req.category)[:40],
            "hits": int(req.hits or 0),
            "titles": [str(t)[:120] for t in (req.titles or [])][:10],
            "kb_error": str(req.kb_error)[:300],
            "ms": int(req.ms or 0),
            "model": str(req.model)[:60],
            "answer": str(req.answer)[:4000],
            "error": str(req.error)[:500],
            "vote": str(req.vote)[:8],
            "intent": str(req.intent)[:500],
        }
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"ok": True}
    except Exception as e:
        logger.warning("log_event failed: %s", e)
        return {"ok": False, "error": str(e)}


@app.get("/api/guard")
def guard_check(q: str = ""):
    """提问侧硬拦截查询：只判断是否命中拦截规则，不调模型。

    供释放域助手页（plan-explainer）在提问前调用：
    - 命中拦截 → blocked=true + reply，前端直接展示服务器下发的固定话术；
    - 未命中   → kb_only=true，前端必须先调 /api/kb 检索，再按 gate 决定能否调模型。

    gate{miss,off} 是知识库闸门未命中时的两段兜底话术。
    """
    from guard import score_hits, _load_rules

    q = (q or "").strip()
    blocked, reply, category = check_question(q)

    best, threshold, gate_ok = 0.0, 0.25, False
    try:
        hits = kb.search(q, top_k=RECALL_K) if q else []
        scored = score_hits(q, hits) if hits else []
        best = scored[0][1] if scored else 0.0
        try:
            threshold = float(_load_rules().get("检索命中阈值", threshold))
        except Exception:
            pass
        gate_ok = best >= threshold
    except Exception:
        pass

    gap = knowledge_gap_reply()
    return {
        "blocked": bool(blocked),
        "category": category,
        "reply": reply,
        # 未被拦截 → 前端不得直接调模型：必须先 /api/kb 检索，并按 gate 判定
        "kb_only": (not blocked),
        "gate": {
            "on": True,
            "min": threshold,
            "miss": gap,          # 领域内但知识库没收录
            "off": gap,           # 完全不相关
            "relevance": best,
            "gate_ok": gate_ok,
        },
    }


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/guard")
def guard_admin():
    """拦截规则可视化管理页。"""
    return FileResponse(str(FRONTEND_DIR / "guard_admin.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config["server"]["host"], port=config["server"]["port"])
