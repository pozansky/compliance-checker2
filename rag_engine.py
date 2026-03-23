import os
import re
import json
import math
from typing import Dict, Any, List, Tuple
import warnings
import logging
import httpx

# 关闭所有 LangChain 相关警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 关闭 LangChain 控制台追踪
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGCHAIN_VERBOSE", "false")
# 关闭 LangChain 控制台追踪，避免打印检索结果等中间步骤

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ComplianceRAGEngine:
    NEGATION_TERMS: Tuple[str, ...] = ("不能", "不要", "别", "不得", "禁止", "不可", "严禁")
    # 按产品类型生效的事件：事件名称 -> 仅在对应 product_type 时保留（"1.0"/"2.0"/"3.0"）
    PRODUCT_TYPE_GATED_EVENTS: Dict[str, str] = {
        "虚假宣传案例精选及人工推票": "1.0",
        "冒用沈杨老师名义": "2.0",
        "对投研调研活动夸大宣传": "3.0",
        "夸大宣传策略重仓操作": "3.0",
    }
    # 规则 ID -> 规则名称（与 prompt 白名单、product_type 过滤共用）
    RULE_NAMES: Dict[int, str] = {
        1: "直接承诺收益",
        2: "突出客户盈利反馈",
        3: "突出描述个股涨幅绩效",
        4: "对投研调研活动夸大宣传",
        5: "向客户索要手机号",
        6: "使用敏感词汇",
        7: "异常开户",
        8: "干扰风险测评独立性",
        9: "错误表述服务合同生效起始周期",
        10: "不文明用语",
        11: "以退款为营销卖点",
        12: "怂恿客户使用他人身份办理服务",
        13: "违规指导",
        14: "将具体股票策略接入权限作为即时办理卖点",
        15: "虚假宣传案例精选及人工推票",
        16: "冒用沈杨老师名义",
        17: "收受客户礼品",
        18: "夸大宣传策略重仓操作",
        19: "虚假宣传",
        20: "变相承诺收益",
    }

    def __init__(
        self,
        retrieve_k: int = None,
        retrieve_score_threshold: float = None,
        max_rules: int = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        # 检索与分块参数（可由调用方传入或从环境变量读取，便于调参）
        def _int_env(name: str, default: int) -> int:
            v = os.getenv(name)
            return int(v) if v is not None and v.strip() != "" else default

        def _float_env(name: str, default: float) -> float:
            v = os.getenv(name)
            return float(v) if v is not None and v.strip() != "" else default

        self._retrieve_k = retrieve_k if retrieve_k is not None else _int_env("RAG_RETRIEVE_K", 20)
        self._retrieve_score_threshold = (
            retrieve_score_threshold
            if retrieve_score_threshold is not None
            else _float_env("RAG_RETRIEVE_SCORE_THRESHOLD", 0.35)
        )
        self._max_rules = max_rules if max_rules is not None else _int_env("RAG_MAX_RULES", 6)
        self._chunk_size = chunk_size if chunk_size is not None else _int_env("RAG_CHUNK_SIZE", 600)
        self._chunk_overlap = chunk_overlap if chunk_overlap is not None else _int_env("RAG_CHUNK_OVERLAP", 200)

        # 最近一次 LLM 请求的 request_id（便于日志追踪）
        self._last_request_id: str | None = None
        self._rule_name_to_id: Dict[str, int] = {v: k for k, v in self.RULE_NAMES.items()}

        # 1. 初始化嵌入模型 (使用本地模型确保语义匹配精度)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # 2. 构建规则向量库（使用分块策略提高召回率）
        self._initialize_vector_store()

        # 2.1 构建 good/bad case 向量库（从 src/cases/*.md 读取）
        self._initialize_case_store()

        # 3. 初始化 LLM (保持原有参数以确保确定性)
        # 为了记录 DashScope 的 request_id，这里通过 httpx 客户端拦截底层 HTTP 响应头。

        def _dashscope_response_hook(response: httpx.Response) -> None:
            try:
                rid = (
                    response.headers.get("x-request-id")
                    or response.headers.get("X-Request-Id")
                    or response.headers.get("request-id")
                )
                if rid:
                    # 直接打到 processing.log（根 logger 会接管），方便排查
                    logger.info(f"DashScope HTTP 响应 request_id={rid}")
            except Exception:
                # 不因日志问题影响主流程
                pass

        # 固定使用你提供的 DashScope key，保证与语音转写一致
        os.environ["DASHSCOPE_API_KEY"] = "sk-eb015732b43844a7980f0daf9eba556d"

        self._http_client = httpx.Client(event_hooks={"response": [_dashscope_response_hook]})

        self.llm = ChatOpenAI(
            model="qwen3.5-plus",
            openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.0,
            max_tokens=800,
            top_p=1.0,
            seed=42,
            max_retries=2,
            request_timeout=60,
            http_client=self._http_client,
            # 千问思考模式：须通过 extra_body 传入，否则会报 Completions.create() unexpected keyword 'enable_thinking'
            extra_body={"enable_thinking": False},
        )

        # 4. 定义 RAG 专用 Prompt（评分版：规则细节在 rules.md → {context}，此处只保留角色、约束、阈值、输出格式）
        _prompt_raw = """
你是证券投顾场景的【合规风险评分引擎】。根据下方【规则背景】中的候选规则做加减分，得到 0–100 的 risk_score，并输出 decision（violation/review/compliant）与 confidence。

【必须遵守】
1. 必须基于规则中的明确文字证据，不得因“整体感觉”直接判 violation。
2. 同一文本可同时含多种风险因子与保护因子，须综合加减分，不得“命中一句就违规”。
3. 保护因子（风险提示、历史说明、免责声明等）命中时须减分。

【判定原则（最高优先级）】
- 以短句为单位分析，严禁跨句组合解读；一个“保证”只对其所在短句负责；“保证”后接服务/价格/流程/一致性等非收益内容时，该句合规。
- 诱导、暗示、营销话术（如“信我”“好吗？”“可以吗？”）不单独判违规；未出现“保证/一定/肯定”+具体金钱收益的死线结构时，不判违规。
- 严格字面匹配：规则中写明的“绝对排除”“一律合规”情形必须执行；禁止联想、加戏、过度解读。

【规则背景】以下为与待检测文本相关的候选规则（每条均为加减分制，分值以规则内为准）：

{context}

【bad/good 示例】仅作类比参考，结合当前文本独立判断：

{cases_context}

【事件名称白名单】rule_name 只能从下列中选择：

__EVENT_WHITELIST__

【决策阈值】risk_score 决定 decision（与系统一致）：
- risk_score ≥ 30 → "violation"；15 ≤ risk_score < 30 → "review"；risk_score < 15 → "compliant"。
confidence：证据清晰 0.8~1.0，有模糊或保护较多 0.5~0.8，仅低风险或模糊 0.0~0.5。

【输出格式】仅输出一个 JSON 对象，字段（大小写一致）：
- risk_score：0–100 数值
- decision："violation"|"review"|"compliant"
- confidence：0–1
- risk_factors：[{{ "rule_id", "rule_name"（来自白名单）, "level":"high"|"medium"|"low", "weight"（正数）, "sentence" }}]
- protective_factors：[{{ "rule_id"（可选）, "rule_name"（可选）, "weight"（负数）, "sentence" }}]
- summary_reason：简洁中文说明
要求：仅此 JSON，无多余文字；无明显风险时 risk_score≈0、decision="compliant"。

【规则13一致性约束（仅限“违规指导”）】
- 如果你在 summary_reason 中明确写出“构成违规指导”“符合规则13”“属于规则13高风险”等结论，则 risk_factors 必须包含 rule_id=13 / rule_name="违规指导"，且 decision 不能为 "compliant"。
- 如果你判断“不构成违规指导”“不符合规则13”“属于服务介绍/功能教学/状态确认/风险提醒”等非违规场景，则 risk_factors 中不得包含 rule_id=13 / rule_name="违规指导"，且 summary_reason 不得再写“构成违规指导”“符合规则13”之类相反结论。
- 严禁出现：summary_reason 说“构成违规指导”，但 risk_factors/decision 却显示未触发；或 summary_reason 说“不构成违规指导”，但 risk_factors 却仍包含“违规指导”。

【待检测内容】

{input}

请依据【规则背景】与上述格式，只输出一个 JSON 对象。
"""
        _event_whitelist = "\n".join(f"{i}. {self.RULE_NAMES[i]}、" for i in range(1, 21)).rstrip("、")
        prompt = ChatPromptTemplate.from_template(_prompt_raw.replace("__EVENT_WHITELIST__", _event_whitelist))

        # 5. 检索配置：按规则召回完整规则，不 rerank（参数已在 __init__ 开头从 kwargs/env 设置）
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self._retrieve_k})
        self.retriever_with_score = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": self._retrieve_k, "score_threshold": self._retrieve_score_threshold}
        )

        # 6. 构建主链路：规则上下文 + case 示例 + 原始文本 → LLM → JSON 字符串
        self.chain = (
            {
                "context": RunnableLambda(self._retrieve_rules_full) | RunnableLambda(self._format_docs),
                "cases_context": RunnableLambda(self._build_cases_context),
                "input": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | RunnableLambda(self._capture_and_parse_llm_output)
        )

    def _format_docs(self, docs):
        """格式化检索到的文档"""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def _capture_and_parse_llm_output(self, llm_output: Any) -> str:
        """
        保存本次 LLM 调用的 request_id，并返回纯文本内容。

        ChatOpenAI.invoke 一般返回 AIMessage，response_metadata 中会带有 request_id。
        """
        # 提取 request_id 供日志追踪
        try:
            meta = getattr(llm_output, "response_metadata", {}) or {}
            self._last_request_id = meta.get("request_id") or meta.get("request-id")
        except Exception:
            self._last_request_id = None

        # 提取文本内容
        try:
            content = llm_output.content  # AIMessage
        except Exception:
            content = str(llm_output)

        # 记录到日志（仅在存在 request_id 时）
        if self._last_request_id:
            logger.info(f"ComplianceRAGEngine LLM 调用完成，request_id={self._last_request_id}")

        return content

    # ==============================
    # case 向量库相关
    # ==============================

    def _initialize_case_store(self):
        """从 src/cases 下的 per-event markdown 中构建 good/bad case 向量库。"""
        from glob import glob

        self.case_vectorstore = None
        self._case_docs: List[Document] = []
        self._rule_calibration: Dict[int, str] = {}  # rule_id -> 易错说明（人工校准提示）
        self._good_case_texts_by_rule: Dict[int, List[str]] = {}
        self._good_case_embeddings_by_rule: Dict[int, List[List[float]]] = {}
        self._bad_case_texts_by_rule: Dict[int, List[str]] = {}
        self._bad_case_embeddings_by_rule: Dict[int, List[List[float]]] = {}
        self._calibration_hints: Dict[int, Dict[str, List[str]]] = {}
        self._structured_calibration_rules: Dict[int, List[Dict[str, Any]]] = {}

        base_dir = os.path.dirname(os.path.abspath(__file__))
        cases_dir = os.path.join(base_dir, "cases")
        if not os.path.isdir(cases_dir):
            return

        pattern = os.path.join(cases_dir, "E??_*.md")
        file_paths = sorted(glob(pattern))
        if not file_paths:
            return

        docs: List[Document] = []

        for path in file_paths:
            filename = os.path.basename(path)
            # 解析规则 ID（E01_E02...）
            m = re.match(r"E(\d{2})_.*\.md", filename)
            if not m:
                continue
            try:
                rule_id = int(m.group(1))
            except ValueError:
                continue
            rule_name = self._get_rule_name_by_id(rule_id)

            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue

            # 简单按 markdown 小标题解析 Definition / Bad cases / Good cases / 易错说明
            sections = self._split_case_markdown(content)
            definition = sections.get("definition", "").strip()
            bad_cases = sections.get("bad cases", [])
            good_cases = sections.get("good cases", [])
            calibration = (sections.get("calibration") or "").strip()
            if calibration:
                self._rule_calibration[rule_id] = calibration
                self._calibration_hints[rule_id] = self._parse_calibration_hints(calibration)
            calibration_rules_text = (sections.get("calibration rules") or "").strip()
            if calibration_rules_text:
                parsed_rules = self._parse_structured_calibration_rules(calibration_rules_text)
                if parsed_rules:
                    self._structured_calibration_rules[rule_id] = parsed_rules
            elif calibration:
                auto_rules = self._compile_calibration_rules_from_text(calibration)
                if auto_rules:
                    self._structured_calibration_rules[rule_id] = auto_rules

            if definition:
                docs.append(
                    Document(
                        page_content=f"[DEFINITION][规则{rule_id}:{rule_name}] {definition}",
                        metadata={
                            "rule_id": rule_id,
                            "rule_name": rule_name,
                            "case_type": "definition",
                            "file": filename,
                        },
                    )
                )

            for text in bad_cases:
                text = text.strip()
                if not text:
                    continue
                self._bad_case_texts_by_rule.setdefault(rule_id, []).append(text)
                docs.append(
                    Document(
                        page_content=f"[BAD][规则{rule_id}:{rule_name}] {text}",
                        metadata={
                            "rule_id": rule_id,
                            "rule_name": rule_name,
                            "case_type": "bad",
                            "file": filename,
                        },
                    )
                )

            for text in good_cases:
                text = text.strip()
                if not text:
                    continue
                self._good_case_texts_by_rule.setdefault(rule_id, []).append(text)
                docs.append(
                    Document(
                        page_content=f"[GOOD][规则{rule_id}:{rule_name}] {text}",
                        metadata={
                            "rule_id": rule_id,
                            "rule_name": rule_name,
                            "case_type": "good",
                            "file": filename,
                        },
                    )
                )

        if not docs:
            return

        self._case_docs = docs
        self.case_vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.case_retriever = self.case_vectorstore.as_retriever(search_kwargs={"k": 50})

        # 预先缓存 good case 向量，用于“good case 强匹配减分”
        try:
            for rid, texts in self._good_case_texts_by_rule.items():
                if not texts:
                    continue
                vectors = self.embeddings.embed_documents(texts)
                if vectors:
                    self._good_case_embeddings_by_rule[rid] = vectors
            for rid, texts in self._bad_case_texts_by_rule.items():
                if not texts:
                    continue
                vectors = self.embeddings.embed_documents(texts)
                if vectors:
                    self._bad_case_embeddings_by_rule[rid] = vectors
        except Exception as e:
            logger.warning(f"初始化 case 向量失败: {e}")
            self._good_case_embeddings_by_rule = {}
            self._bad_case_embeddings_by_rule = {}

    def _split_case_markdown(self, content: str) -> Dict[str, Any]:
        """
        解析单个 case markdown：
        期望小标题为：
        - ## Definition
        - ## Bad cases
        - ## Good cases
        返回：
        {
          "definition": "xxx",
          "bad cases": ["句1", "句2"],
          "good cases": ["句A", "句B"]
        }
        """
        lines = content.splitlines()
        sections: Dict[str, List[str]] = {}
        current_key = None

        for line in lines:
            header_match = re.match(r"^##\s+(.*)", line.strip())
            if header_match:
                header = header_match.group(1).strip().lower()
                if header.startswith("definition"):
                    current_key = "definition"
                elif header.startswith("bad"):
                    current_key = "bad cases"
                elif header.startswith("good cases"):
                    current_key = "good cases"
                elif header.startswith("good"):
                    # 兼容只写 Good 的情况
                    current_key = "good cases"
                elif header.startswith("risk keywords"):
                    current_key = "risk keywords"
                elif header.startswith("strong protection keywords"):
                    current_key = "strong protection keywords"
                elif header.startswith("calibration rules") or header.startswith("校准规则"):
                    current_key = "calibration rules"
                elif header.startswith("易错说明") or header.startswith("calibration"):
                    current_key = "calibration"
                else:
                    current_key = None
                if current_key not in sections:
                    sections[current_key] = []
                continue

            if current_key:
                sections.setdefault(current_key, []).append(line)

        result: Dict[str, Any] = {}
        if "definition" in sections:
            result["definition"] = "\n".join(sections["definition"]).strip()

        def _extract_list(items: List[str]) -> List[str]:
            out: List[str] = []
            for l in items:
                s = l.strip()
                if not s:
                    continue
                # markdown 列表项 "- xxx"
                s = re.sub(r"^[-*+]\s*", "", s)
                if s:
                    out.append(s)
            return out

        if "bad cases" in sections:
            result["bad cases"] = _extract_list(sections["bad cases"])
        if "good cases" in sections:
            result["good cases"] = _extract_list(sections["good cases"])
        if "risk keywords" in sections:
            result["risk keywords"] = _extract_list(sections["risk keywords"])
        if "strong protection keywords" in sections:
            result["strong protection keywords"] = _extract_list(sections["strong protection keywords"])
        if "calibration" in sections:
            result["calibration"] = "\n".join(sections["calibration"]).strip()
        if "calibration rules" in sections:
            result["calibration rules"] = "\n".join(sections["calibration rules"]).strip()

        return result

    def _retrieve_case_examples(self, text: str, candidate_rule_ids: List[int], k_per_rule: int = 3) -> Dict[int, Dict[str, List[str]]]:
        """
        在 case 向量库中检索与文本最相似的 bad/good case。
        返回结构：
        {
          rule_id: {
            "bad": [句1, 句2],
            "good": [句A, 句B],
          },
          ...
        }
        """
        result: Dict[int, Dict[str, List[str]]] = {}
        if not getattr(self, "case_vectorstore", None) or not candidate_rule_ids:
            return result

        text = self._normalize_input_text(text)
        # 先全局检索一批，再按 rule_id 过滤聚合
        try:
            docs = self.case_retriever.invoke(text)
        except Exception:
            docs = []

        for doc in docs:
            rid = doc.metadata.get("rule_id")
            if rid not in candidate_rule_ids:
                continue
            case_type = doc.metadata.get("case_type", "")
            if case_type not in ("bad", "good"):
                continue
            # 去掉前缀标签，仅保留原始句子
            content = doc.page_content
            content = re.sub(r"^\[[A-Z]+\]\[[^\]]+\]\s*", "", content).strip()
            if not content:
                continue

            bucket = result.setdefault(rid, {"bad": [], "good": []})
            lst_key = "bad" if case_type == "bad" else "good"
            if len(bucket[lst_key]) >= k_per_rule:
                continue
            if content not in bucket[lst_key]:
                bucket[lst_key].append(content)

        return result

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """计算两个向量的余弦相似度。"""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _best_good_case_match(self, text_embedding: List[float], rule_id: int, input_text: str = "") -> Tuple[float, str]:
        """
        返回某条规则下，输入文本与 good case 的最高相似度及命中的 good case 文本。
        若无可用 good case，返回 (0.0, "")。
        """
        vectors = self._good_case_embeddings_by_rule.get(rule_id) or []
        texts = self._good_case_texts_by_rule.get(rule_id) or []
        if not text_embedding or not vectors or not texts:
            return 0.0, ""

        best_sim = 0.0
        best_text = ""
        for idx, vec in enumerate(vectors):
            candidate = texts[idx] if idx < len(texts) else ""
            if not self._polarity_consistent(input_text, candidate):
                continue
            sim = self._cosine_similarity(text_embedding, vec)
            if sim > best_sim:
                best_sim = sim
                best_text = candidate
        return best_sim, best_text

    def _best_bad_case_match(self, text_embedding: List[float], rule_id: int, input_text: str = "") -> Tuple[float, str]:
        """
        返回某条规则下，输入文本与 bad case 的最高相似度及命中的 bad case 文本。
        若无可用 bad case，返回 (0.0, "")。
        """
        vectors = self._bad_case_embeddings_by_rule.get(rule_id) or []
        texts = self._bad_case_texts_by_rule.get(rule_id) or []
        if not text_embedding or not vectors or not texts:
            return 0.0, ""

        best_sim = 0.0
        best_text = ""
        for idx, vec in enumerate(vectors):
            candidate = texts[idx] if idx < len(texts) else ""
            if not self._polarity_consistent(input_text, candidate):
                continue
            sim = self._cosine_similarity(text_embedding, vec)
            if sim > best_sim:
                best_sim = sim
                best_text = candidate
        return best_sim, best_text

    def _normalize_case_text(self, s: str) -> str:
        """用于 bad/good case 字面匹配：去空白和常见标点，转小写。"""
        if not s:
            return ""
        s = s.lower().strip()
        s = re.sub(r"[\s，。！？!?,；;：“”\"'、\(\)\[\]【】]", "", s)
        return s

    def _contains_negation(self, s: str) -> bool:
        s = s or ""
        return any(t in s for t in self.NEGATION_TERMS)

    def _polarity_consistent(self, text: str, case_text: str) -> bool:
        """否定极性一致性：一方是否定、另一方非否定，则判为不一致。"""
        return self._contains_negation(text) == self._contains_negation(case_text)

    def _literal_case_match(self, text: str, case_text: str) -> bool:
        """字面匹配：case 是否包含于文本或文本包含于 case。"""
        if not self._polarity_consistent(text, case_text):
            return False
        a = self._normalize_case_text(text)
        b = self._normalize_case_text(case_text)
        if not a or not b:
            return False
        return b in a or a in b

    def _is_risk_assessment_context(self, text: str) -> bool:
        """Detect questionnaire/risk-assessment workflow context."""
        if not text:
            return False
        patterns = [
            r"风险测评|风测|测评|问卷|风险问卷",
            r"第\s*\d+\s*题|A\s*[、\.．)]?\s*|B\s*[、\.．)]?\s*|C\s*[、\.．)]?\s*|D\s*[、\.．)]?\s*",
            r"及格|通过|重测|重新测评|测评分",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _has_assessment_steering(self, text: str) -> bool:
        """Detect explicit steering/tampering in risk assessment context."""
        if not text:
            return False
        if not self._is_risk_assessment_context(text):
            return False

        strong_patterns = [
            r"我说几号你就选几号|按我说的填|照着我说的填|听我选",
            r"选\s*[ABCD]\s*才(能)?过|别选(保守|低风险)|必须选\s*[ABCD]",
            r"往高了填|改高一点|做高一点|填高一点",
            r"这样才能过|保证通过|包过|一定通过",
            r"身份证(随便填|乱填|填别人)|姓名(随便填|乱填|填别人)",
        ]
        if any(re.search(p, text, re.IGNORECASE) for p in strong_patterns):
            return True

        # Soft guidance alone is not considered interference.
        soft_only_patterns = [
            r"随便选一个|随便填|都可以|没有标准答案|只要不空着就行",
        ]
        if any(re.search(p, text, re.IGNORECASE) for p in soft_only_patterns):
            return False

        return False

    def _get_e13_service_intro_keywords(self) -> List[str]:
        return [
            "老师", "老师的建议", "特训营", "APP", "app", "微信", "查看", "推送", "具体的股票", "止盈区间",
            "止损区间", "止盈止损区间", "选股器", "二维码",
            "会给你", "会给到", "会告诉你", "会推送", "会提示", "会通知",
            "合作后", "加入后", "加入合作之后","合作之后","根据安排来", "内部", "系统会", "软件会", "老师会",
            "盘中会", "统一推送", "会员", "服务期", "体验期", "带你", "跟上",
            "都会给到", "都会发", "内部会员", "内部服务", "办理", "办理通道",
            "明确的代码", "明确的股票代码", "什么价位买", "什么价位时间卖", "价格区间去买", "价格区间去卖出", "直接指导",
            "投资顾问", "执业编号", "个人观点", "不构成您投资的依据", "不构成投资依据", "风险自担", "独立决策",
            "助教", "老师助教", "专业老师", "专业老师助教", "发送到你手机", "发送到手机", "手机里提示", "发到手机",
            "解套方案", "服务范围之内", "票质量怎么样", "加入进来之后",
            "买卖点位", "指导建议", "微信通知", "发微信通知", "电话再提醒", "打个电话再提醒", "走申请", "带你办理",
            "跟上老师的步伐", "图片里面写", "写的很详细", "确保一定不会错过","特许营","黄金杯", "严选量投","研选量投", "内部策略", "高级助教", "模型选股", "股池", "系统", "软件", "模型案例",
        "趋势智投","私人定制","三步点金","解套王","解套王指标","机构跟投","投研跟投","案例精选","金牌客服","服务顾问","选股魔方",
        "主力先锋","龙回头","顶底拐点","热点雷达","资金龙虎榜","操盘线","十全十美","波段擒牛","底部强力","超跌精灵","底分拐点","模型精选",
        "缴费", "续费", "优惠名额", "锁定优惠名额", "服务时间", "赠送服务时间","什么时间方便缴费",
            "带您走后面的流程", "已经申请好优惠", "申请锁定了优惠名额", "正好98000", "98000",
            "服务续费", "服务售卖", "服务跟进", "优惠套餐", "办理续费", "办理缴费",
            "先交定金", "定金", "尾款", "补尾款", "信用卡", "整理账户", "先申请下来"]

    def _analyze_e13_context(self, text: str, summary_reason: str) -> Dict[str, Any]:
        text = text or ""
        summary_reason = summary_reason or ""
        summary_reason_norm = re.sub(r"[\s“”\"'‘’《》〈〉「」『』（）()\[\]【】,，。；;：:、】【]", "", summary_reason)

        service_intro_keywords = self._get_e13_service_intro_keywords()
        status_check_keywords = [
            "没买股票", "没买", "为什么没买", "怎么没买", "卖掉了吗", "是不是卖掉了",
            "是不是清了", "是不是清仓了", "你这个不是卖掉了吗", "今天是什么原因没买股票",
            "持仓状态", "确认持仓", "看下持仓", "亏大了", "为什么没进",
            "持有的什么票", "持仓什么票", "现在你是持有的什么票", "你现在拿的什么票", "现在拿的什么票"
        ]
        risk_reminder_keywords = [
            "止盈止损", "设置好严格的止盈止损", "设置好止盈止损", "控制仓位", "避免重仓",
            "不建议重仓", "不要重仓", "别重仓", "轻仓参与", "仓位控制", "避免被套",
            "风险分散", "分散风险", "设置止损", "设置止盈"
        ]
        education_keywords = [
            "圆弧底", "洗盘", "杯子", "上涨形态", "形态教学", "二次下砸确认", "主力资金",
            "主力资金接入", "政策利好方向", "提前释放风险", "形态", "走势", "技术形态",
            "投教", "教学", "复盘", "策略教学"
        ]
        product_keywords = ["特许营","黄金杯", "严选量投","研选量投", "内部策略", "高级助教", "模型选股", "股池", "系统", "软件", "模型案例",
        "趋势智投","私人定制","三步点金","解套王","机构跟投","投研跟投","案例精选","金牌客服","服务顾问","选股魔方",
        "主力先锋","龙回头","顶底拐点","热点雷达","资金龙虎榜","操盘线","十全十美","波段擒牛","底部强力","超跌精灵","底分拐点","解套王指标"]
        action_keywords = ["买入", "卖出", "加仓", "减仓", "持有", "清仓", "止损", "止盈", "做T", "建仓", "平仓"]
        strong_action_keywords = ["买入", "卖出", "加仓", "减仓", "持有", "清仓", "止损", "止盈", "建仓", "平仓", "离场", "兑现", "出手", "博一搏",
                                    "买进", "卖掉", "割肉", "补仓", "满仓", "半仓",
                                    "轻仓", "重仓", "空仓", "进场", "出局", "撤掉",
                                    "拿住", "拿着", "落袋", "冲高出", "低吸", "高抛",
                                    "抄底", "逃顶", "开仓", "斩仓", "锁仓",
                                    "全部卖出", "全部买入", "全部清掉", "全部减掉", "全部离场",
                                    "马上买", "赶紧卖", "立刻进", "直接卖", "直接买",
                                    "现在买", "现在卖", "现在清", "现在走",
                                    "轻仓进", "出掉一些","走掉"]
        action_patterns = [
            r"买入", r"卖出", r"加仓", r"减仓", r"持有", r"清仓", r"止损", r"止盈", r"做T", r"建仓", r"平仓",
            r"离场", r"减仓避险", r"降低持仓成本", r"小仓位试错", r"加仓跟进",
            r"等待企稳信号", r"关注", r"博一搏", r"试试", r"尝试"
        ]
        strong_action_patterns = [
            r"买入", r"卖出", r"加仓", r"减仓", r"持有", r"清仓", r"止损", r"止盈", r"建仓", r"平仓",
            r"离场", r"减仓避险", r"降低持仓成本", r"小仓位试错", r"加仓跟进", r"兑现", r"出手",
            r"博一搏", r"搏一搏", r"(轻仓|半仓).{0,6}(拿|博)"
        ]
        condition_keywords = [
            "现在", "今天", "明天", "马上", "立即", "开盘", "收盘",
            "全仓", "半仓", "轻仓", "重仓", "跌破", "突破", "回踩", "止损", "止盈",
            "一半", "一半仓", "减到一半", "三分之一", "1/2", "1/3"
        ]
        condition_patterns = [
            r"现在", r"今天", r"明天", r"马上", r"立即", r"开盘", r"收盘",
            r"全仓", r"半仓", r"轻仓", r"重仓", r"跌破", r"突破", r"回踩",
            r"\d+(?:\.\d+)?元", r"\d+(?:\.\d+)?块", r"\d+(?:\.\d+)?~\d+(?:\.\d+)?元",
            r"\d+(?:\.\d+)?-\d+(?:\.\d+)?元", r"站稳", r"企稳", r"放量突破",
            r"目标看至", r"逢高", r"下方", r"上方", r"附近",
            r"一半", r"减到一半", r"减仓一半", r"半仓", r"三分之一", r"1/2", r"1/3"
        ]
        negative_summary_phrases = [
            "不符合规则 13", "不符合规则13", "不构成违规指导", "不属于违规指导",
            "不符合违规指导", "不触发规则 13", "不触发规则13", "不触发违规指导",
            "不触发规则 13（违规指导）", "不触发规则13（违规指导）", "不触发规则13违规指导",
            "未涉及证券投资建议", "未涉及任何证券投资建议", "未涉及具体股票标的", "未涉及交易指令",
            "未涉及收益承诺", "未涉及合同生效时间表述", "未命中规则 13", "未命中规则13",
            "未命中任何候选违规规则", "不命中任何候选违规规则", "不涉及具体股票标的", "不涉及交易指令",
            "无违规风险", "无风险因子", "不涉及证券投资建议",
            "未指定具体个股标的", "未出现具体股票标识", "属于强保护因子",
            "综合判定为合规", "判定为合规", "合规情形", "服务介绍/转述", "服务介绍", "转述",
            "投资顾问", "执业编号", "个人观点", "不构成您投资的依据", "不构成投资依据", "风险自担",
            "故不加分", "缺失规则 13 判定的核心要素", "缺失具体股票标识", "缺失具体股票标的",
            "未构成针对特定个股的当前可执行指令", "未构成当前可执行指令","不命中规则 13（违规指导）"
        ]

        target_exclude_keywords = set(
            service_intro_keywords
            + product_keywords
            + [
                "这个票", "那个票", "这只票", "那只票", "今天一直", "如果说", "可以持有",
                "考虑暂时", "五日线啊", "老师团队", "全程通知", "统一推送", "内部会员",
                "指导提示", "价格区间", "核心区间", "强支撑位", "震荡中枢", "压力位",
                "历史高位", "操作建议", "持仓状态", "空仓状态", "关键观察", "风险提示"
            ]
        )

        has_explicit_target = bool(re.search(r"\d{6}", text))
        if not has_explicit_target:
            stock_name_suffix_pattern = (
                r"[\u4e00-\u9fa5]{2,6}"
                r"(集团|科技|电信|黄金|重工|股份|药业|电子|控股|能源|银行|证券|实业|材料|制造|智能|信息|通信)"
            )
            explicit_name_hits = re.findall(stock_name_suffix_pattern, text)
            if explicit_name_hits:
                has_explicit_target = True

        if not has_explicit_target:
            stock_name_candidates = re.findall(r"[A-Za-z\u4e00-\u9fa5]{4,8}", text)  
            for candidate in stock_name_candidates:
                if candidate in target_exclude_keywords:
                    continue
                if any(keyword in candidate for keyword in target_exclude_keywords):
                    continue
                if candidate.endswith(("老师", "团队", "会员", "策略", "软件", "系统")):
                    continue
                if len(candidate) == 4 and any("\u4e00" <= ch <= "\u9fa5" for ch in candidate):
                    has_explicit_target = True
                    break

        price_point_hits = re.findall(r"\d+(?:\.\d+)?(?:元|块)", text)
        implicit_target_signal = (
            len(price_point_hits) >= 2
            and any(token in text for token in ["这个票", "这只票", "那个票", "那只票", "它", "他"])
        )
        has_target = has_explicit_target or implicit_target_signal
        has_action = any(keyword in text for keyword in action_keywords) or any(re.search(pattern, text) for pattern in action_patterns)
        has_strong_action = any(keyword in text for keyword in strong_action_keywords) or any(re.search(pattern, text) for pattern in strong_action_patterns)
        has_condition = any(keyword in text for keyword in condition_keywords) or any(re.search(pattern, text) for pattern in condition_patterns)
        contains_service_intro = any(keyword in text for keyword in service_intro_keywords)
        contains_product_name = any(keyword in text for keyword in product_keywords)
        is_status_check = any(keyword in text for keyword in status_check_keywords)
        is_risk_reminder = any(keyword in text for keyword in risk_reminder_keywords)
        is_education_talk = any(keyword in text for keyword in education_keywords)
        is_capability_sale = (
            (
                "内部服务" in text
                or "办理" in text
                or "办理通道" in text
                or "合作之后" in text
                or "加入合作之后" in text
            )
            and (
                "明确的代码" in text
                or "明确的股票代码" in text
                or "什么价位买" in text
                or "什么价位时间卖" in text
                or "价格区间去买" in text
                or "价格区间去卖出" in text
                or "直接指导" in text
                or "根据安排来" in text
            )
        )
        if (
            "加入特许营" in text
            and "老师团队全程通知" in text
            and "跟着买跟着卖就好" in text
        ):
            contains_service_intro = True
        if is_status_check or is_risk_reminder or is_education_talk:
            contains_service_intro = True
        negative_summary_phrases_norm = [re.sub(r"[\s“”\"'‘’《》〈〉「」『』（）()\[\]【】,，。；;：:、】【]", "", phrase) for phrase in negative_summary_phrases]
        summary_negates_e13 = any(
            phrase in summary_reason or phrase_norm in summary_reason_norm
            for phrase, phrase_norm in zip(negative_summary_phrases, negative_summary_phrases_norm)
        )
        summary_supports_e13 = (
            not summary_negates_e13
            and (
                ((("规则 13" in summary_reason or "规则13" in summary_reason) and ("违规指导" in summary_reason or "构成违规指导" in summary_reason)))
                or ("构成违规指导" in summary_reason or "构成违规指导" in summary_reason_norm)
                or ("符合规则 13" in summary_reason or "符合规则13" in summary_reason or "符合规则13" in summary_reason_norm)
                or ("判定为违规指导" in summary_reason or "判定为违规指导" in summary_reason_norm)
                or ("完全符合规则 13" in summary_reason or "完全符合规则13" in summary_reason or "完全符合规则13" in summary_reason_norm)
                or ("高风险违规指导特征" in summary_reason or "高风险违规指导特征" in summary_reason_norm)
            )
        )
        rule_supports_e13 = (
            has_target
            and has_strong_action
            and has_condition
            and not contains_service_intro
            and not contains_product_name
            and not is_capability_sale
        )
        loose_rule_supports_e13 = (
            has_action
            and has_condition
            and not contains_service_intro
            and not contains_product_name
            and not is_capability_sale
        )
        should_block_e13 = (
            contains_service_intro
            or contains_product_name
            or is_capability_sale
            or (not has_target and not implicit_target_signal)
            or not has_strong_action
            or not has_condition
        )

        return {
            "summary_supports_e13": summary_supports_e13,
            "summary_negates_e13": summary_negates_e13,
            "rule_supports_e13": rule_supports_e13,
            "loose_rule_supports_e13": loose_rule_supports_e13,
            "should_block_e13": should_block_e13,
            "contains_service_intro": contains_service_intro,
            "contains_product_name": contains_product_name,
            "is_status_check": is_status_check,
            "is_risk_reminder": is_risk_reminder,
            "is_education_talk": is_education_talk,
            "has_target": has_target,
            "has_action": has_action,
            "has_strong_action": has_strong_action,
            "has_condition": has_condition,
        }

    def _analyze_e01_context(self, text: str, summary_reason: str) -> Dict[str, Any]:
        text = text or ""
        summary_reason = summary_reason or ""
        sentences = [s.strip() for s in re.split(r"[。！？!?；;\n|]", text) if s.strip()]

        promise_terms = ["保证", "一定", "肯定", "肯定要","绝对", "稳赚", "必赚", "包赚", "保本", "包赔", "可以", "能够", "能"]
        result_terms = ["赚钱", "盈利", "收益", "获利", "回本", "不亏", "翻倍", "赚回来"]
        result_patterns = [
            r"\d+\s*个点",
            r"\d+(?:\.\d+)?\s*%",
            r"翻倍",
        ]
        strong_protection_terms = ["投资有风险", "不保证收益", "不承诺盈利", "不代表未来表现", "不构成收益保证"]
        negative_summary_terms = ["不构成直接承诺收益", "不属于直接承诺收益", "未承诺收益", "未承诺盈利", "客户自述", "历史盈利事实"]

        has_same_sentence_promise_and_result = False
        for sent in sentences:
            has_promise = any(term in sent for term in promise_terms)
            has_result = any(term in sent for term in result_terms) or any(re.search(pattern, sent) for pattern in result_patterns)
            if has_promise and has_result:
                has_same_sentence_promise_and_result = True
                break

        has_strong_protection = any(term in text for term in strong_protection_terms)
        summary_negates_e01 = any(term in summary_reason for term in negative_summary_terms)
        summary_supports_e01 = (
            ("直接承诺收益" in summary_reason or "规则 1" in summary_reason or "规则1" in summary_reason)
            and not summary_negates_e01
        )
        rule_supports_e01 = has_same_sentence_promise_and_result
        should_block_e01 = (not rule_supports_e01) or has_strong_protection

        return {
            "summary_supports_e01": summary_supports_e01,
            "summary_negates_e01": summary_negates_e01,
            "rule_supports_e01": rule_supports_e01,
            "should_block_e01": should_block_e01,
            "has_strong_protection": has_strong_protection,
        }

    def _parse_structured_calibration_rules(self, text: str) -> List[Dict[str, Any]]:
        """
        解析结构化校准规则（JSON 数组）。
        支持直接 JSON 或 ```json fenced block```。
        """
        if not text:
            return []
        raw = text.strip()
        fence_match = re.search(r"```json\s*(.*?)\s*```", raw, re.IGNORECASE | re.DOTALL)
        if fence_match:
            raw = fence_match.group(1).strip()
        try:
            obj = json.loads(raw)
        except Exception:
            return []
        if not isinstance(obj, list):
            return []
        out: List[Dict[str, Any]] = []
        for item in obj:
            if isinstance(item, dict) and isinstance(item.get("type"), str):
                out.append(item)
        return out

    def _compile_calibration_rules_from_text(self, calibration_text: str) -> List[Dict[str, Any]]:
        """
        从“易错说明”自然语言自动编译基础结构化规则。
        若未手写 Calibration Rules，可直接使用该编译结果。
        """
        if not calibration_text:
            return []

        lines = [l.strip() for l in calibration_text.splitlines() if l.strip()]
        compiled: List[Dict[str, Any]] = []
        numeric_regex = [r"\d+\s*%", r"\d+\s*万", r"\d+\s*块", r"每股", r"一股"]

        for line in lines:
            rule_type = None
            if "易误判" in line:
                rule_type = "false_positive"
            elif "易漏判" in line:
                rule_type = "false_negative"
            if not rule_type:
                continue

            quoted_terms = re.findall(r"[“\"]([^”\"]+)[”\"]", line)
            quoted_terms = [q.strip() for q in quoted_terms if q.strip()]
            rule: Dict[str, Any] = {
                "type": rule_type,
                "note": "auto_compiled_from_calibration",
            }
            if quoted_terms:
                rule["any"] = quoted_terms

            has_no_numeric = ("无具体金额" in line) or ("无具体盈利数字" in line) or ("未提盈利数据" in line)
            has_numeric_required = ("具体盈利数字" in line) or ("金额/百分比" in line) or ("金额、百分比" in line)

            if rule_type == "false_positive" and has_no_numeric:
                rule["not_regex"] = numeric_regex
            if rule_type == "false_negative" and has_numeric_required:
                rule["any_regex"] = numeric_regex

            if "客户/案例" in line or ("客户" in line and "案例" in line):
                rule["any"] = list(dict.fromkeys((rule.get("any") or []) + ["客户", "案例"]))

            if any(k in rule for k in ("any", "all", "any_regex", "all_regex", "not_any", "not_regex")):
                compiled.append(rule)

        return compiled

    def _matches_structured_rule(self, text: str, rule: Dict[str, Any]) -> bool:
        text = text or ""

        def _as_list(v: Any) -> List[str]:
            if isinstance(v, list):
                return [str(x) for x in v if str(x).strip()]
            return []

        any_terms = _as_list(rule.get("any"))
        all_terms = _as_list(rule.get("all"))
        not_any_terms = _as_list(rule.get("not_any"))
        any_regex = _as_list(rule.get("any_regex"))
        all_regex = _as_list(rule.get("all_regex"))
        not_regex = _as_list(rule.get("not_regex"))

        if any_terms and not any(t in text for t in any_terms):
            return False
        if all_terms and not all(t in text for t in all_terms):
            return False
        if not_any_terms and any(t in text for t in not_any_terms):
            return False
        if any_regex and not any(re.search(p, text) for p in any_regex):
            return False
        if all_regex and not all(re.search(p, text) for p in all_regex):
            return False
        if not_regex and any(re.search(p, text) for p in not_regex):
            return False
        return True

    def _parse_calibration_hints(self, calibration_text: str) -> Dict[str, List[str]]:
        """
        从“易错说明”中抽取可规则化短语：
        - false_positive: 易误判（应减分/降级）
        - false_negative: 易漏判（应加分/提级）
        """
        hints: Dict[str, List[str]] = {"false_positive": [], "false_negative": []}
        if not calibration_text:
            return hints

        lines = [l.strip() for l in calibration_text.splitlines() if l.strip()]
        for line in lines:
            bucket = None
            if "易误判" in line:
                bucket = "false_positive"
            elif "易漏判" in line:
                bucket = "false_negative"
            if bucket is None:
                continue

            # 提取中文引号内短语，作为可匹配提示词
            quoted = re.findall(r"[“\"]([^”\"]+)[”\"]", line)
            for q in quoted:
                s = (q or "").strip()
                if s and s not in hints[bucket]:
                    hints[bucket].append(s)
        return hints

    def _has_hard_risk_pattern(self, text: str, rule_id: int) -> bool:
        """
        硬风险兜底：命中硬风险结构时，不允许 good case 覆盖。
        当前先实现规则1（直接承诺收益）的硬风险识别。
        """
        if rule_id != 1:
            return False

        guarantee_re = re.compile(r"(保证|保本|稳赚|一定|肯定|绝对|承诺)")
        profit_re = re.compile(r"(赚钱|盈利|收益|获利|翻倍|涨停)")
        sentences = re.split(r"[。！？!?；;\n]", text or "")
        for sent in sentences:
            s = sent.strip()
            if not s:
                continue
            if guarantee_re.search(s) and profit_re.search(s):
                return True
        return False

    def _has_official_abctougu_addwx_link(self, text: str) -> bool:
        """
        Official addwx links are allowed service/onboarding links and should not trigger E07.
        """
        t = (text or "").lower()
        if not t:
            return False
        return (
            "abctougu.com/addwx/index" in t
            or "crm.abctougu.cn/addwx/index" in t
        )

    def _has_service_payment_onboarding(self, text: str) -> bool:
        """
        Service/payment onboarding flow should not be treated as E07 abnormal account opening.
        """
        t = (text or "").lower()
        if not t:
            return False
        keywords = [
            "支付宝", "扫码", "扫一扫", "二维码", "保存到相册", "相册", "截图", "扫一扫",
            "支付", "付款", "支付方式", "付款页", "支付页", "开通", "开通服务", "办理", "通道",
            "内部通道", "服务", "合作后", "合作之后", "加入之后", "加入合作之后", "内部服务",
            "手机", "手机里", "联系客服"
        ]
        return any(keyword.lower() in t for keyword in keywords)

    def _build_cases_context(self, text: Any) -> str:
        """根据候选规则 ID，从 case 库中选出若干典型 bad/good case 供模型类比参考。"""
        text_norm = self._normalize_input_text(text)
        candidate_ids = self._get_candidate_rule_ids(text_norm)
        if not candidate_ids:
            return "（未检索到明显相关的案例，可按文本本身独立判断。）"

        # 限制进入上游 Prompt 的候选规则数，避免上下文过长
        max_rules_for_cases = 4
        candidate_ids = candidate_ids[:max_rules_for_cases]

        case_map = self._retrieve_case_examples(text_norm, candidate_ids, k_per_rule=3)
        if not case_map:
            return "（未检索到明显相关的案例，可按文本本身独立判断。）"

        parts: List[str] = []
        for rid in candidate_ids:
            if rid not in case_map and rid not in getattr(self, "_rule_calibration", {}):
                continue
            rule_name = self._get_rule_name_by_id(rid)
            buckets = case_map.get(rid) or {}
            parts.append(f"【规则{rid}: {rule_name}】")
            bad_list = buckets.get("bad") or []
            good_list = buckets.get("good") or []
            if bad_list:
                parts.append("  - 典型 BAD cases：")
                for s in bad_list:
                    parts.append(f"    - [BAD] {s}")
            if good_list:
                parts.append("  - 典型 GOOD cases：")
                for s in good_list:
                    parts.append(f"    - [GOOD] {s}")
            calibration = getattr(self, "_rule_calibration", {}).get(rid)
            if calibration:
                parts.append(f"  - 易错提示（人工校准）：{calibration}")
            parts.append("")

        return "\n".join(parts).strip() or "（未检索到明显相关的案例，可按文本本身独立判断。）"

    def _normalize_input_text(self, text: Any) -> str:
        """保证检索/预测入口拿到的是字符串，避免 dict/None 导致 TypeError。"""
        if text is None:
            return ""
        if isinstance(text, str):
            return text
        if isinstance(text, dict):
            return str(text.get("input", text.get("context", "")) or "")
        return str(text)

    def _get_candidate_rule_ids(self, text: str) -> List[int]:
        """用分块语义检索 + 关键词匹配得到候选规则 ID 列表（去重、保序）。"""
        text = self._normalize_input_text(text)
        seen: set = set()
        ordered: List[int] = []
        try:
            semantic_docs = self.retriever_with_score.invoke(text)
            # 若在设定的相似度阈值下完全检索不到规则块，记录一条带阈值的日志，方便后续评估阈值是否合理
            if not semantic_docs:
                logger.info(
                    f"RAG 语义检索无结果，当前 score_threshold={self._retrieve_score_threshold}; "
                    f"后续将回退到普通相似度检索并结合关键词匹配。"
                )
        except Exception:
            semantic_docs = self.retriever.invoke(text)
        for doc in semantic_docs:
            rid = doc.metadata.get("rule_id")
            if rid is not None and rid not in seen:
                seen.add(rid)
                ordered.append(rid)
        for rule_id, _score, _kw in self._keyword_match_rules(text):
            if rule_id not in seen:
                seen.add(rule_id)
                ordered.append(rule_id)
        return ordered

    def _retrieve_rules_full(self, text: str) -> List[Document]:
        """按规则召回：先得到候选规则 ID，再取每条规则的完整原文，按候选顺序取 top N 条（不做 rerank）。"""
        text = self._normalize_input_text(text)
        candidate_ids = self._get_candidate_rule_ids(text)
        candidates: List[Tuple[int, str]] = []
        for rid in candidate_ids:
            full_text = self._full_rules_by_id.get(rid)
            if full_text:
                candidates.append((rid, full_text))
        if not candidates:
            return []
        candidates = candidates[: self._max_rules]
        return [
            Document(
                page_content=full_text,
                metadata={"rule_id": rid, "rule_name": self._get_rule_name_by_id(rid)}
            )
            for rid, full_text in candidates
        ]

    def _retrieve_hybrid(self, text: str) -> List[Document]:
        """混合检索（分块）：仅用于调试对比；主流程已改为 _retrieve_rules_full。"""
        try:
            semantic_docs = self.retriever_with_score.invoke(text)
        except Exception:
            semantic_docs = self.retriever.invoke(text)
        seen_chunks = {(doc.metadata.get("rule_id"), doc.metadata.get("chunk_type", "")) for doc in semantic_docs}
        result = list(semantic_docs)
        for rule_id, _score, _kw in self._keyword_match_rules(text)[:5]:
            for doc in self._rule_id_to_docs.get(rule_id, []):
                key = (doc.metadata.get("rule_id"), doc.metadata.get("chunk_type", ""))
                if key not in seen_chunks:
                    seen_chunks.add(key)
                    result.append(doc)
        return result[:20] if len(result) > 20 else result

    def _initialize_vector_store(self):
        """将完整的20条规则分块存储到向量库；若存在未过期的本地 FAISS 缓存则直接加载，否则构建并落盘。"""
        _src_dir = os.path.dirname(os.path.abspath(__file__))
        rules_path = os.path.join(_src_dir, "rules.md")
        index_dir = os.path.join(_src_dir, "faiss_index")
        meta_file = os.path.join(index_dir, "meta.txt")

        def _build_full_rules_by_id_and_keyword():
            full_rules = self._get_full_rules_content()
            self._full_rules_by_id = {i + 1: full_rules[i] for i in range(len(full_rules))}
            self._build_rule_keyword_index()

        # 尝试从本地缓存加载（需 rules.md 未变更）
        if os.path.isdir(index_dir) and os.path.isfile(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    saved_mtime = f.read().strip()
                if os.path.isfile(rules_path) and saved_mtime == str(os.path.getmtime(rules_path)):
                    self.vectorstore = FAISS.load_local(
                        index_dir, self.embeddings, allow_dangerous_deserialization=True
                    )
                    _build_full_rules_by_id_and_keyword()
                    self._rule_id_to_docs = {}
                    for doc in getattr(self.vectorstore.docstore, "_dict", {}).values():
                        rid = doc.metadata.get("rule_id")
                        if rid is not None:
                            self._rule_id_to_docs.setdefault(rid, []).append(doc)
                    return
            except Exception:
                pass

        # 构建向量库
        full_rules = self._get_full_rules_content()
        self._full_rules_by_id = {i + 1: full_rules[i] for i in range(len(full_rules))}
        documents = []
        for i, rule_text in enumerate(full_rules):
            rule_id = i + 1
            rule_name = self._get_rule_name_by_id(rule_id)
            chunks = self._split_rule_into_chunks(rule_text, rule_id, rule_name)
            documents.extend(chunks)
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self._rule_id_to_docs = {}
        for doc in documents:
            rid = doc.metadata.get("rule_id")
            if rid is not None:
                self._rule_id_to_docs.setdefault(rid, []).append(doc)
        self._build_rule_keyword_index()

        # 落盘并记录 rules.md mtime
        try:
            os.makedirs(index_dir, exist_ok=True)
            self.vectorstore.save_local(index_dir)
            with open(meta_file, "w", encoding="utf-8") as f:
                f.write(str(os.path.getmtime(rules_path)))
        except Exception:
            pass

    # 规则正文滑动窗口：按「段」建索引，embedding 能命中任意违规点；送入 LLM 仍用 _full_rules_by_id 整条（_chunk_size/_chunk_overlap 在 __init__ 中设置）

    def _split_rule_into_chunks(self, rule_text: str, rule_id: int, rule_name: str) -> List[Document]:
        """将单条规则分割成多个小块：滑动窗口覆盖全文 + 语义段（标题/违规情形/排除条款等）。"""
        chunks = []
        size, overlap = self._chunk_size, self._chunk_overlap
        step = max(1, size - overlap)
        # 滑动窗口覆盖整条规则，便于检索命中任意一段（含后半段排除条款等）
        for i, start in enumerate(range(0, len(rule_text), step)):
            segment = rule_text[start : start + size]
            if not segment.strip():
                continue
            chunks.append(Document(
                page_content=f"【规则{rule_id}: {rule_name}】\n{segment}",
                metadata={"rule_id": rule_id, "rule_name": rule_name, "chunk_type": "window", "chunk_index": i}
            ))
        if not chunks:
            chunks.append(Document(
                page_content=f"【规则{rule_id}: {rule_name}】\n{rule_text}",
                metadata={"rule_id": rule_id, "rule_name": rule_name, "chunk_type": "window", "chunk_index": 0}
            ))

        # 语义段：标题、核心逻辑、违规情形、排除条款、重要说明
        # 1. 提取标题和核心逻辑
        title_match = re.search(r"### \d+\. (.*?)\n", rule_text)
        if title_match:
            title_part = title_match.group(1)
            chunks.append(Document(
                page_content=f"【规则{rule_id}标题】{title_part}",
                metadata={"rule_id": rule_id, "rule_name": rule_name, "chunk_type": "title"}
            ))
        
        # 2. 提取核心逻辑部分（如果有）
        core_logic_pattern = r"【核心逻辑】.*?(?=\n\n|$)"
        core_logic_match = re.search(core_logic_pattern, rule_text, re.DOTALL)
        if core_logic_match:
            core_logic = core_logic_match.group(0)
            chunks.append(Document(
                page_content=f"【规则{rule_id}核心逻辑】{core_logic}",
                metadata={"rule_id": rule_id, "rule_name": rule_name, "chunk_type": "core_logic"}
            ))
        
        # 3. 提取具体违规情形
        violation_pattern = r"具体违规情形.*?(?=绝对排除条款|重要说明|最终判断|$)"
        violation_match = re.search(violation_pattern, rule_text, re.DOTALL | re.IGNORECASE)
        if violation_match:
            violations = violation_match.group(0)
            chunks.append(Document(
                page_content=f"【规则{rule_id}违规情形】{violations}",
                metadata={"rule_id": rule_id, "rule_name": rule_name, "chunk_type": "violation"}
            ))
        
        # 4. 提取绝对排除条款
        exclusion_pattern = r"绝对排除条款.*?(?=\n\n|重要说明|最终判断|$)"
        exclusion_match = re.search(exclusion_pattern, rule_text, re.DOTALL | re.IGNORECASE)
        if exclusion_match:
            exclusions = exclusion_match.group(0)
            chunks.append(Document(
                page_content=f"【规则{rule_id}排除条款】{exclusions}",
                metadata={"rule_id": rule_id, "rule_name": rule_name, "chunk_type": "exclusion"}
            ))
        
        # 5. 提取重要说明
        note_pattern = r"重要说明.*?(?=\n\n|最终判断|$)"
        note_match = re.search(note_pattern, rule_text, re.DOTALL | re.IGNORECASE)
        if note_match:
            notes = note_match.group(0)
            chunks.append(Document(
                page_content=f"【规则{rule_id}重要说明】{notes}",
                metadata={"rule_id": rule_id, "rule_name": rule_name, "chunk_type": "note"}
            ))
        
        return chunks

    def _build_rule_keyword_index(self):
        """从 src/cases/*.md 中构建规则关键词索引，便于人工在 Markdown 中维护。"""
        from glob import glob

        self.rule_keywords = {}
        self.rule_protection_keywords: Dict[int, List[str]] = {}

        base_dir = os.path.dirname(os.path.abspath(__file__))
        cases_dir = os.path.join(base_dir, "cases")
        if not os.path.isdir(cases_dir):
            return

        pattern = os.path.join(cases_dir, "E??_*.md")
        for path in sorted(glob(pattern)):
            filename = os.path.basename(path)
            m = re.match(r"E(\d{2})_.*\.md", filename)
            if not m:
                continue
            try:
                rule_id = int(m.group(1))
            except ValueError:
                continue

            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue

            sections = self._split_case_markdown(content)
            risk_keywords = sections.get("risk keywords", []) or []
            strong_protection_keywords = sections.get("strong protection keywords", []) or []

            # 用于触发候选规则的关键词：只使用风险侧关键词，避免保护性词汇拉高风险匹配分
            cleaned_risk = [k.strip().lower() for k in risk_keywords if k.strip()]
            if cleaned_risk:
                self.rule_keywords[rule_id] = cleaned_risk

            cleaned_protect = [k.strip().lower() for k in strong_protection_keywords if k.strip()]
            if cleaned_protect:
                self.rule_protection_keywords[rule_id] = cleaned_protect

    def _get_rule_name_by_id(self, rule_id: int) -> str:
        """根据规则ID获取规则名称"""
        return self.RULE_NAMES.get(rule_id, f"规则{rule_id}")

    def _get_full_rules_content(self) -> List[str]:
        """
        从外部 Markdown 文件 `rules.md` 读取并返回 20 条完整规则内容。
        这样你只需要编辑 Markdown 文件即可维护规则文本。
        """
        rules_path = os.path.join(os.path.dirname(__file__), "rules.md")
        if not os.path.exists(rules_path):
            raise FileNotFoundError(f"规则文件不存在: {rules_path}")

        with open(rules_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 按顶层规则标题 `### 1. ...` / `### 2. ...` 切分
        lines = content.splitlines(keepends=True)
        rule_blocks: List[str] = []
        current: List[str] = []

        for line in lines:
            # 匹配形如 "### 1. 标题" 的规则标题；已开始收集内容时遇到新的标题则开始新规则
            if re.match(r"^###\s+\d+\.", line) and current:
                rule_blocks.append("".join(current).strip())
                current = [line]
            else:
                current.append(line)

        if current:
            rule_blocks.append("".join(current).strip())

        # 只保留以 "### 数字. 标题" 开头的规则块（排除文件开头的使用说明等）
        rule_blocks = [blk for blk in rule_blocks if blk.strip() and re.match(r"^###\s+\d+\.", blk.lstrip())]

        if len(rule_blocks) != 20:
            raise ValueError(
                f"解析到的规则数量为 {len(rule_blocks)}，预期为 20，请检查 rules.md 中顶层标题是否为 '### 序号. 标题' 格式。"
            )

        return rule_blocks

    def _keyword_match_rules(self, text: str) -> List[Tuple[int, int, List[str]]]:
        """Keyword matching fallback. Returns (rule_id, score, matched_keywords)."""
        if not isinstance(text, str):
            text = self._normalize_input_text(text)
        text_lower = (text or "").lower()
        text_compact = self._normalize_case_text(text_lower)
        matches = []

        for rule_id, keywords in self.rule_keywords.items():
            score = 0
            matched_keywords = []

            for keyword in keywords:
                keyword_lower = (keyword or "").lower()
                keyword_compact = self._normalize_case_text(keyword_lower)
                # Normal match + compact match (space/punctuation split tolerant).
                hit = (keyword_lower in text_lower) or (
                    bool(keyword_compact) and keyword_compact in text_compact
                )
                if hit:
                    score += 1
                    matched_keywords.append(keyword)

            if score > 0:
                matches.append((rule_id, score, matched_keywords))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def predict(self, text: str, product_type=None, _is_direct_promise_retry: bool = False) -> Dict[str, Any]:
        """预测违规情况（加减分 + 阈值）：按规则识别风险因子/保护因子并累加得到 risk_score，再按阈值判违规。

        返回风险分、decision、confidence、触发事件及理由等；是否违规由 risk_score 与 violation_threshold 决定，非“命中即违规”。

        product_type: 产品类型。仅影响四个事件的生效范围：
        - "虚假宣传案例精选及人工推票" 仅在 product_type 为 1.0 时触发；
        - "冒用沈杨老师名义" 仅在 product_type 为 2.0 时触发；
        - "对投研调研活动夸大宣传"、"夸大宣传策略重仓操作" 仅在 product_type 为 3.0 时触发；
        - 其他事件与未传 product_type 时均为全量检测。
        """
        # 阈值可通过环境变量微调，方便线上校准
        def _float_env(name: str, default: float) -> float:
            v = os.getenv(name)
            try:
                return float(v) if v is not None and v.strip() != "" else default
            except Exception:
                return default

        # 加减分制阈值：≥30 违规，15~30 人工复核，<15 合规；可通过环境变量覆盖
        violation_threshold = _float_env("RISK_VIOLATION_THRESHOLD", 30.0)
        review_threshold = _float_env("RISK_REVIEW_THRESHOLD", 15.0)
        good_case_override_threshold = _float_env("GOOD_CASE_OVERRIDE_THRESHOLD", 0.82)
        good_case_override_ratio = _float_env("GOOD_CASE_OVERRIDE_RATIO", 1.0)
        good_case_force_discount = _float_env("GOOD_CASE_FORCE_DISCOUNT", 20.0)
        bad_case_match_threshold = _float_env("BAD_CASE_MATCH_THRESHOLD", 0.82)
        bad_case_force_score = _float_env("BAD_CASE_FORCE_SCORE", 35.0)
        bad_case_bonus = _float_env("BAD_CASE_BONUS", 20.0)
        calibration_fp_discount = _float_env("CALIBRATION_FP_DISCOUNT", 15.0)
        calibration_fn_bonus = _float_env("CALIBRATION_FN_BONUS", 20.0)

        try:
            text = self._normalize_input_text(text)
            # 1. 调用 LLM 获取原始响应（预期为 JSON 字符串）
            raw_response = self.chain.invoke(text, config={"callbacks": []})
            if raw_response is None:
                raw_response = ""
            raw_response = str(raw_response).strip()

            # 2. 尝试解析 JSON（容错处理：截取最外层花括号）
            parsed: Dict[str, Any] = {}
            parse_error = None
            try:
                parsed = json.loads(raw_response)
            except Exception as e1:
                parse_error = str(e1)
                try:
                    start = raw_response.find("{")
                    end = raw_response.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        parsed = json.loads(raw_response[start : end + 1])
                        parse_error = None
                except Exception as e2:
                    parse_error = f"{parse_error} | {str(e2)}"

            if not isinstance(parsed, dict):
                raise ValueError(f"无法解析模型 JSON 响应: {parse_error or '未知错误'}")

            # 3. 读取基础字段（带默认值）
            risk_score_raw = parsed.get("risk_score", 0)
            try:
                risk_score = float(risk_score_raw)
            except Exception:
                risk_score = 0.0
            if risk_score < 0:
                risk_score = 0.0
            if risk_score > 100:
                risk_score = 100.0

            decision = str(parsed.get("decision", "") or "").strip().lower()
            confidence_raw = parsed.get("confidence", 0.0)
            try:
                confidence = float(confidence_raw)
            except Exception:
                confidence = 0.0
            if confidence < 0:
                confidence = 0.0
            if confidence > 1:
                confidence = 1.0

            risk_factors = parsed.get("risk_factors") or []
            protective_factors = parsed.get("protective_factors") or []
            summary_reason = parsed.get("summary_reason") or ""

            # 4. 根据阈值与 decision 得到布尔违规结果
            violation_by_score = risk_score >= violation_threshold
            violation = decision == "violation" or violation_by_score

            # 5. 从 risk_factors 中抽取触发事件及理由，并累计每个事件的风险得分
            triggered_events: List[str] = []
            event_reasons: Dict[str, str] = {}
            event_scores: Dict[str, float] = {}

            def _join_triggered_events() -> str:
                return ", ".join(triggered_events) if triggered_events else "无"

            def _decision_from_score() -> str:
                return (
                    "violation"
                    if risk_score >= violation_threshold
                    else ("review" if risk_score >= review_threshold else "compliant")
                )

            def _remove_event(rule_name: str) -> float:
                nonlocal risk_score
                removed_score = float(event_scores.pop(rule_name, 0.0) or 0.0)
                triggered_events[:] = [e for e in triggered_events if e != rule_name]
                event_reasons.pop(rule_name, None)
                if removed_score > 0:
                    risk_score = max(0.0, risk_score - removed_score)
                return removed_score

            def _upsert_event(rule_name: str, score: float, reason: str) -> None:
                nonlocal risk_score
                old_score = float(event_scores.get(rule_name, 0.0) or 0.0)
                new_score = float(score)
                if old_score > 0:
                    risk_score = max(0.0, risk_score - old_score)
                event_scores[rule_name] = new_score
                risk_score = min(100.0, risk_score + new_score)
                if rule_name not in triggered_events:
                    triggered_events.append(rule_name)
                if reason:
                    event_reasons[rule_name] = reason

            if isinstance(risk_factors, list):
                for factor in risk_factors:
                    if not isinstance(factor, dict):
                        continue
                    rule_name = str(factor.get("rule_name") or "").strip()
                    if not rule_name:
                        continue
                    weight = factor.get("weight", 0)
                    try:
                        weight_val = float(weight)
                    except Exception:
                        weight_val = 0.0
                    # 只把正向风险因子当作“触发事件”
                    if weight_val <= 0:
                        continue
                    event_scores[rule_name] = event_scores.get(rule_name, 0.0) + weight_val
                    sentence = str(factor.get("sentence") or "").strip()
                    if rule_name not in triggered_events:
                        triggered_events.append(rule_name)
                    # 若该事件还没有理由，则记录一句代表性文本
                    if rule_name not in event_reasons and sentence:
                        event_reasons[rule_name] = sentence

            triggered_event_str = ", ".join(triggered_events) if triggered_events else "无"

            # 6. 按 product_type 过滤特定事件（保持与旧版逻辑一致）
            original_violation = violation
            original_triggered_event_str = triggered_event_str
            original_event_reasons = event_reasons.copy()
            pt = product_type
            if pt is not None:
                if pt in (1, "1", "1.0"):
                    pt = "1.0"
                elif pt in (2, "2", "2.0"):
                    pt = "2.0"
                elif pt in (3, "3", "3.0"):
                    pt = "3.0"
                else:
                    pt = None

            if pt is not None and triggered_event_str != "无":
                normalized = re.sub(r"[，、;；\s]+", ",", triggered_event_str)
                events = [e.strip() for e in normalized.split(",") if e.strip()]
                filtered_events: List[str] = []
                for e in events:
                    keep = True
                    for gated_name, required_pt in self.PRODUCT_TYPE_GATED_EVENTS.items():
                        if gated_name in e and pt != required_pt:
                            keep = False
                            break
                    if keep:
                        filtered_events.append(e)
                filtered_events = list(dict.fromkeys(filtered_events))

                filtered_event_reasons: Dict[str, str] = {}
                filtered_event_scores: Dict[str, float] = {}
                for evt_name, evt_reason in event_reasons.items():
                    keep = True
                    for gated_name, required_pt in self.PRODUCT_TYPE_GATED_EVENTS.items():
                        if gated_name in evt_name and pt != required_pt:
                            keep = False
                            break
                    if keep:
                        filtered_event_reasons[evt_name] = evt_reason
                        if evt_name in event_scores:
                            filtered_event_scores[evt_name] = event_scores[evt_name]

                triggered_event_str = ", ".join(filtered_events) if filtered_events else "无"
                event_reasons = filtered_event_reasons
                event_scores = filtered_event_scores

                # 若按产品类型过滤后没有剩余高风险事件，则可以视情况下调违规结论
                if not filtered_events:
                    # 若 risk_score 主要来源于被过滤事件，理论上应降低违规等级。
                    # 为了安全，这里仅在 risk_score 不高时（小于 violation 阈值）自动降为合规。
                    if risk_score < violation_threshold:
                        violation = False

            # 6.4 E13 最小化后处理：只校正“违规指导”本身，不影响其他事件
            e13_name = self._get_rule_name_by_id(13)
            if e13_name:
                e13_ctx = self._analyze_e13_context(text, summary_reason)
                summary_supports_e13 = bool(e13_ctx["summary_supports_e13"])
                summary_negates_e13 = bool(e13_ctx["summary_negates_e13"])
                rule_supports_e13 = bool(e13_ctx["rule_supports_e13"])
                should_block_e13 = bool(e13_ctx["should_block_e13"])

                if e13_name in event_scores:
                    # 先尊重大模型命中结果，再用规则只校正 E13 自身，防止误判。
                    if should_block_e13:
                        _remove_event(e13_name)
                        triggered_event_str = _join_triggered_events()
                        decision = _decision_from_score()
                        violation = decision == "violation"
                    elif summary_negates_e13 and not rule_supports_e13:
                        _remove_event(e13_name)
                        triggered_event_str = _join_triggered_events()
                        decision = _decision_from_score()
                        violation = decision == "violation"
                    elif summary_supports_e13:
                        _upsert_event(e13_name, event_scores.get(e13_name, 60.0) or 60.0, str(summary_reason).strip())
                        triggered_event_str = _join_triggered_events()
                        decision = _decision_from_score()
                        violation = decision == "violation"
                else:
                    # E13 规则只做后置过滤，不再承担补漏判。
                    pass

            # 6.4 bad case 强匹配触发：支持“未触发也可因 bad case 命中触发”
            bad_case_hits: List[Dict[str, Any]] = []
            if getattr(self, "_bad_case_embeddings_by_rule", None):
                try:
                    text_embedding = self.embeddings.embed_query(text)
                except Exception:
                    text_embedding = []

                if text_embedding:
                    candidate_rule_ids = self._get_candidate_rule_ids(text)
                    # 对字面直匹配，放宽到全量 bad case 规则，避免候选召回漏掉该规则。
                    literal_hit_rule_ids = set()
                    for rid_all, bad_texts in (self._bad_case_texts_by_rule or {}).items():
                        for bad_text in bad_texts:
                            if self._literal_case_match(text, bad_text):
                                literal_hit_rule_ids.add(rid_all)
                                break
                    rule_ids_for_bad_match = list(dict.fromkeys(candidate_rule_ids + list(literal_hit_rule_ids)))
                    for rid in rule_ids_for_bad_match:
                        best_sim, best_bad_case = self._best_bad_case_match(text_embedding, rid, text)
                        # 字面直匹配优先：命中则按 sim=1.0 处理，避免相似度阈值漏掉近似同句。
                        for bad_text in (self._bad_case_texts_by_rule.get(rid) or []):
                            if self._literal_case_match(text, bad_text):
                                best_sim = 1.0
                                best_bad_case = bad_text
                                break
                        if best_sim < bad_case_match_threshold:
                            continue
                        rule_name = self._get_rule_name_by_id(rid)
                        if self._has_hard_risk_pattern(text, rid):
                            delta = max(0.0, bad_case_bonus)
                        elif rule_name in event_scores:
                            delta = max(0.0, bad_case_bonus)
                        else:
                            delta = max(0.0, bad_case_force_score)
                        if delta <= 0:
                            continue

                        risk_score = min(100.0, risk_score + delta)
                        event_scores[rule_name] = event_scores.get(rule_name, 0.0) + delta
                        if rule_name not in triggered_events:
                            triggered_events.append(rule_name)
                        if rule_name not in event_reasons:
                            event_reasons[rule_name] = best_bad_case or "命中 BAD case 强匹配"
                        bad_case_hits.append(
                            {
                                "rule_id": rid,
                                "rule_name": rule_name,
                                "bad_case_score": round(best_sim, 4),
                                "matched_bad_case": best_bad_case,
                                "bonus": round(delta, 2),
                            }
                        )
                    triggered_event_str = ", ".join(triggered_events) if triggered_events else "无"
            else:
                bad_case_hits = []

            # 6.5 good case 强匹配覆盖：对命中的事件做减分（保留硬风险兜底）
            good_case_overrides: List[Dict[str, Any]] = []
            override_total = 0.0
            if getattr(self, "_good_case_embeddings_by_rule", None):
                if 'text_embedding' not in locals():
                    try:
                        text_embedding = self.embeddings.embed_query(text)
                    except Exception:
                        text_embedding = []

                # 与 bad case 一致：字面直匹配可以绕过候选召回限制。
                literal_good_hit_rule_ids = set()
                for rid_all, good_texts in (self._good_case_texts_by_rule or {}).items():
                    for good_text in good_texts:
                        if self._literal_case_match(text, good_text):
                            literal_good_hit_rule_ids.add(rid_all)
                            break
                event_rule_ids = [self._rule_name_to_id.get(name) for name in event_scores.keys()]
                event_rule_ids = [rid for rid in event_rule_ids if rid is not None]
                rule_ids_for_good_match = list(dict.fromkeys(event_rule_ids + list(literal_good_hit_rule_ids)))

                removed_events = set()
                for rid in rule_ids_for_good_match:
                    evt_name = self._get_rule_name_by_id(rid)
                    evt_score = float(event_scores.get(evt_name, 0.0) or 0.0)
                    if self._has_hard_risk_pattern(text, rid):
                        continue

                    best_sim, best_good_case = self._best_good_case_match(text_embedding, rid, text)
                    # 字面直匹配优先
                    for good_text in (self._good_case_texts_by_rule.get(rid) or []):
                        if self._literal_case_match(text, good_text):
                            best_sim = 1.0
                            best_good_case = good_text
                            break
                    if best_sim < good_case_override_threshold:
                        continue

                    if evt_score > 0:
                        discount = min(evt_score, max(0.0, evt_score * good_case_override_ratio))
                    else:
                        discount = min(max(0.0, good_case_force_discount), risk_score)
                    if discount <= 0:
                        continue

                    override_total += discount
                    if evt_name in event_scores:
                        removed_events.add(evt_name)
                    good_case_overrides.append(
                        {
                            "rule_id": rid,
                            "rule_name": evt_name,
                            "good_case_score": round(best_sim, 4),
                            "matched_good_case": best_good_case,
                            "discount": round(discount, 2),
                        }
                    )

                if removed_events:
                    risk_score = max(0.0, risk_score - override_total)
                    triggered_events = [e for e in triggered_events if e not in removed_events]
                    triggered_event_str = ", ".join(triggered_events) if triggered_events else "无"
                    event_reasons = {k: v for k, v in event_reasons.items() if k not in removed_events}
                    event_scores = {k: v for k, v in event_scores.items() if k not in removed_events}

            # 6.6 易错说明校准：易误判减分、易漏判加分（按短语命中）
            calibration_hits: List[Dict[str, Any]] = []
            text_norm = text or ""
            structured_calibrated_rule_ids: set[int] = set()

            # 先执行结构化校准规则（优先级高于文本短语校准）
            for rid, rules in (getattr(self, "_structured_calibration_rules", {}) or {}).items():
                if not isinstance(rules, list):
                    continue
                rule_name = self._get_rule_name_by_id(rid)
                for rule_obj in rules:
                    if not isinstance(rule_obj, dict):
                        continue
                    if not self._matches_structured_rule(text_norm, rule_obj):
                        continue
                    rule_type = str(rule_obj.get("type", "") or "").strip().lower()
                    weight = rule_obj.get("weight")
                    try:
                        weight_val = float(weight) if weight is not None else None
                    except Exception:
                        weight_val = None
                    note = str(rule_obj.get("note", "") or "").strip()

                    if (
                        rule_type == "false_positive"
                        and rule_name in event_scores
                        and not self._has_hard_risk_pattern(text_norm, rid)
                    ):
                        delta_base = calibration_fp_discount if weight_val is None else abs(weight_val)
                        current_evt_score = event_scores.get(rule_name, delta_base)
                        delta = min(max(0.0, delta_base), current_evt_score)
                        if delta > 0:
                            risk_score = max(0.0, risk_score - delta)
                            if rule_name in event_scores:
                                new_score = max(0.0, event_scores.get(rule_name, 0.0) - delta)
                                if new_score <= 0.0:
                                    event_scores.pop(rule_name, None)
                                    event_reasons.pop(rule_name, None)
                                    triggered_events = [e for e in triggered_events if e != rule_name]
                                else:
                                    event_scores[rule_name] = new_score
                            calibration_hits.append(
                                {
                                    "rule_id": rid,
                                    "rule_name": rule_name,
                                    "type": "false_positive",
                                    "matched_terms": [note] if note else ["structured_rule"],
                                    "delta": round(-delta, 2),
                                }
                            )
                            structured_calibrated_rule_ids.add(rid)

                    elif rule_type == "false_negative":
                        delta_base = calibration_fn_bonus if weight_val is None else abs(weight_val)
                        delta = max(0.0, delta_base)
                        if delta > 0:
                            risk_score = min(100.0, risk_score + delta)
                            event_scores[rule_name] = event_scores.get(rule_name, 0.0) + delta
                            if rule_name not in triggered_events:
                                triggered_events.append(rule_name)
                            if rule_name not in event_reasons:
                                event_reasons[rule_name] = note or "命中结构化易漏判校准"
                            calibration_hits.append(
                                {
                                    "rule_id": rid,
                                    "rule_name": rule_name,
                                    "type": "false_negative",
                                    "matched_terms": [note] if note else ["structured_rule"],
                                    "delta": round(delta, 2),
                                }
                            )
                            structured_calibrated_rule_ids.add(rid)

            # 再执行旧版“易错说明短语”校准（对已走结构化规则的 rule_id 不重复执行）
            for rid, hint_map in (getattr(self, "_calibration_hints", {}) or {}).items():
                if rid in structured_calibrated_rule_ids:
                    continue
                rule_name = self._get_rule_name_by_id(rid)
                fp_terms = hint_map.get("false_positive") or []
                fn_terms = hint_map.get("false_negative") or []
                matched_fp = [t for t in fp_terms if t and t in text_norm]
                matched_fn = [t for t in fn_terms if t and t in text_norm]

                # 易误判：命中且无硬风险时减分
                if matched_fp and rule_name in event_scores and not self._has_hard_risk_pattern(text_norm, rid):
                    delta = min(max(0.0, calibration_fp_discount), event_scores.get(rule_name, 0.0) if rule_name in event_scores else calibration_fp_discount)
                    if delta > 0:
                        risk_score = max(0.0, risk_score - delta)
                        if rule_name in event_scores:
                            new_score = max(0.0, event_scores.get(rule_name, 0.0) - delta)
                            if new_score <= 0.0:
                                event_scores.pop(rule_name, None)
                                event_reasons.pop(rule_name, None)
                                triggered_events = [e for e in triggered_events if e != rule_name]
                            else:
                                event_scores[rule_name] = new_score
                        calibration_hits.append(
                            {
                                "rule_id": rid,
                                "rule_name": rule_name,
                                "type": "false_positive",
                                "matched_terms": matched_fp,
                                "delta": round(-delta, 2),
                            }
                        )

                # 易漏判：命中时加分
                if matched_fn:
                    delta = max(0.0, calibration_fn_bonus)
                    if delta > 0:
                        risk_score = min(100.0, risk_score + delta)
                        event_scores[rule_name] = event_scores.get(rule_name, 0.0) + delta
                        if rule_name not in triggered_events:
                            triggered_events.append(rule_name)
                        if rule_name not in event_reasons:
                            event_reasons[rule_name] = "命中易漏判校准短语"
                        calibration_hits.append(
                            {
                                "rule_id": rid,
                                "rule_name": rule_name,
                                "type": "false_negative",
                                "matched_terms": matched_fn,
                                "delta": round(delta, 2),
                            }
                        )
            triggered_event_str = ", ".join(triggered_events) if triggered_events else "无"

            # 如果没有任何触发事件，但 risk_score 仍然很高，reason 使用 summary_reason 兜底
            # 6.65 E01 后置过滤：只做“明确反向/明确保护”的误删清理，不做补召回，不影响其他事件
            e01_name = self._get_rule_name_by_id(1)
            if e01_name:
                e01_ctx = self._analyze_e01_context(text_norm, summary_reason)
                summary_supports_e01 = bool(e01_ctx["summary_supports_e01"])
                summary_negates_e01 = bool(e01_ctx["summary_negates_e01"])
                has_strong_protection = bool(e01_ctx["has_strong_protection"])
                should_block_e01 = summary_negates_e01 or has_strong_protection

                if e01_name in event_scores and should_block_e01:
                    _remove_event(e01_name)
                    triggered_event_str = _join_triggered_events()

            # 6.7 conflict resolution: in risk-assessment context, prefer E08 over E13
            e08_name = self._get_rule_name_by_id(8)
            e13_name = self._get_rule_name_by_id(13)
            if (
                self._is_risk_assessment_context(text_norm)
                and self._has_assessment_steering(text_norm)
                and e13_name in event_scores
            ):
                moved_score = float(event_scores.get(e13_name, 0.0) or 0.0)
                event_scores.pop(e13_name, None)
                triggered_events = [e for e in triggered_events if e != e13_name]
                event_reasons.pop(e13_name, None)

                if moved_score > 0:
                    event_scores[e08_name] = event_scores.get(e08_name, 0.0) + moved_score
                    if e08_name not in triggered_events:
                        triggered_events.append(e08_name)
                    if e08_name not in event_reasons:
                        event_reasons[e08_name] = "在风险测评填写中存在替选答案/引导作答，干扰风险测评独立性。"

                bad_case_hits = [h for h in bad_case_hits if str(h.get("rule_name", "")) != e13_name]
                calibration_hits = [h for h in calibration_hits if str(h.get("rule_name", "")) != e13_name]
                good_case_overrides = [h for h in good_case_overrides if str(h.get("rule_name", "")) != e13_name]
                triggered_event_str = ", ".join(triggered_events) if triggered_events else "无"

            # 6.75 conflict resolution: in risk-assessment context, E07 should not stand;
            # if there is explicit steering, map to E08, otherwise clear E07.
            e07_name = self._get_rule_name_by_id(7)
            if self._is_risk_assessment_context(text_norm) and e07_name in event_scores:
                moved_score = float(event_scores.get(e07_name, 0.0) or 0.0)
                event_scores.pop(e07_name, None)
                triggered_events = [e for e in triggered_events if e != e07_name]
                event_reasons.pop(e07_name, None)

                if moved_score > 0 and self._has_assessment_steering(text_norm):
                    event_scores[e08_name] = event_scores.get(e08_name, 0.0) + moved_score
                    if e08_name not in triggered_events:
                        triggered_events.append(e08_name)
                    if e08_name not in event_reasons:
                        event_reasons[e08_name] = "在风测场景中存在明确作答干预，按E08处理。"
                else:
                    risk_score = max(0.0, risk_score - moved_score)

                bad_case_hits = [h for h in bad_case_hits if str(h.get("rule_name", "")) != e07_name]
                calibration_hits = [h for h in calibration_hits if str(h.get("rule_name", "")) != e07_name]
                good_case_overrides = [h for h in good_case_overrides if str(h.get("rule_name", "")) != e07_name]
                triggered_event_str = ", ".join(triggered_events) if triggered_events else "无"

            # 6.8 official addwx whitelist: do not treat official abctougu addwx links as E07 abnormal account opening
            if self._has_official_abctougu_addwx_link(text_norm) and e07_name in event_scores:
                removed_score = float(event_scores.get(e07_name, 0.0) or 0.0)
                event_scores.pop(e07_name, None)
                triggered_events = [e for e in triggered_events if e != e07_name]
                event_reasons.pop(e07_name, None)
                if removed_score > 0:
                    risk_score = max(0.0, risk_score - removed_score)
                bad_case_hits = [h for h in bad_case_hits if str(h.get("rule_name", "")) != e07_name]
                calibration_hits = [h for h in calibration_hits if str(h.get("rule_name", "")) != e07_name]
                good_case_overrides = [h for h in good_case_overrides if str(h.get("rule_name", "")) != e07_name]
                triggered_event_str = ", ".join(triggered_events) if triggered_events else "无"

            # 6.81 service/payment onboarding whitelist: do not treat payment QR / onboarding flows as E07.
            if self._has_service_payment_onboarding(text_norm) and e07_name in event_scores:
                removed_score = float(event_scores.get(e07_name, 0.0) or 0.0)
                event_scores.pop(e07_name, None)
                triggered_events = [e for e in triggered_events if e != e07_name]
                event_reasons.pop(e07_name, None)
                if removed_score > 0:
                    risk_score = max(0.0, risk_score - removed_score)
                bad_case_hits = [h for h in bad_case_hits if str(h.get("rule_name", "")) != e07_name]
                calibration_hits = [h for h in calibration_hits if str(h.get("rule_name", "")) != e07_name]
                good_case_overrides = [h for h in good_case_overrides if str(h.get("rule_name", "")) != e07_name]
                triggered_event_str = ", ".join(triggered_events) if triggered_events else "无"

            # 6.9 E13 final safety net:
            # if the LLM summary has already clearly concluded "违规指导",
            # restore only E13 itself unless the text is clearly a service/product introduction.
            if e13_name:
                e13_ctx = locals().get("e13_ctx") or self._analyze_e13_context(text_norm, summary_reason)
                summary_supports_e13_final = bool(e13_ctx["summary_supports_e13"])
                summary_negates_e13_final = bool(e13_ctx["summary_negates_e13"])
                late_rule_supports_e13 = bool(e13_ctx["rule_supports_e13"])
                late_loose_rule_supports_e13 = bool(e13_ctx.get("loose_rule_supports_e13"))
                late_should_block_e13 = bool(e13_ctx["should_block_e13"])
                late_has_target = bool(e13_ctx.get("has_target"))
                late_has_action = bool(e13_ctx.get("has_action")) or any(
                    keyword in text_norm for keyword in ["买入", "卖出", "加仓", "减仓", "持有", "清仓", "止损", "止盈", "试错", "跟进"]
                )
                late_has_condition = bool(e13_ctx.get("has_condition")) or any(
                    keyword in text_norm for keyword in ["跌破", "突破", "回踩", "站稳", "一半", "半仓", "轻仓", "重仓", "元", "块"]
                )
                late_summary_explicit_support = summary_supports_e13_final or (
                    not summary_negates_e13_final
                    and "违规指导" in summary_reason
                    and any(token in summary_reason for token in ["构成", "符合规则 13", "符合规则13", "高风险判定标准"])
                )
                late_contains_service_intro = bool(e13_ctx.get("contains_service_intro"))
                late_service_phrase_hit = any(
                    keyword in text_norm for keyword in self._get_e13_service_intro_keywords()
)

                # Minimal E13 consistency correction:
                # only when the LLM summary explicitly supports E13 and the structured E13 rule
                # also strongly supports it, but E13 is still missing from final events.
                if (
                    e13_name not in event_scores
                    and late_summary_explicit_support
                    and not late_contains_service_intro
                    and not late_service_phrase_hit
                    and (
                        late_rule_supports_e13
                        or late_loose_rule_supports_e13
                        or sum([late_has_target, late_has_action, late_has_condition]) >= 2
                    )
                    and not (
                        late_should_block_e13
                        and not late_loose_rule_supports_e13
                        and sum([late_has_target, late_has_action, late_has_condition]) < 2
                    )
                ):
                    _upsert_event(e13_name, 60.0, str(summary_reason).strip())
                    triggered_event_str = _join_triggered_events()

                # Final E13 cleanup for multi-event scenarios:
                # if E13 is still present but the text is actually a service intro / capability sale /
                # or the model explicitly negates E13 without hard rule support, remove only E13.
                late_core_signal_support = sum([late_has_target, late_has_action, late_has_condition]) >= 2
                if e13_name in event_scores and (
                    (
                        (late_should_block_e13 or late_service_phrase_hit)
                        and not late_loose_rule_supports_e13
                        and not late_core_signal_support
                    )
                    or (summary_negates_e13_final and not late_rule_supports_e13 and not late_core_signal_support)
                ):
                    _remove_event(e13_name)
                    triggered_event_str = _join_triggered_events()
            if not event_reasons and summary_reason:
                event_reasons = {"综合说明": str(summary_reason)}

            # 7. 为每个事件理由增加事件得分前缀（若有），并在整体理由中附上总分
            formatted_event_reasons: Dict[str, str] = {}
            for evt_name, evt_reason in event_reasons.items():
                prefix = ""
                if evt_name in event_scores:
                    prefix = f"[该事件得分约 {event_scores[evt_name]:.1f}] "
                formatted_event_reasons[evt_name] = f"{prefix}{evt_reason}"

            final_decision = (
                "violation"
                if risk_score >= violation_threshold
                else ("review" if risk_score >= review_threshold else "compliant")
            )
            violation = final_decision == "violation"
            override_reason_suffix = ""
            if good_case_overrides:
                override_parts: List[str] = []
                for item in good_case_overrides:
                    rule_name = str(item.get("rule_name", "") or "")
                    matched_good_case = str(item.get("matched_good_case", "") or "")
                    sim = item.get("good_case_score", 0.0)
                    discount = item.get("discount", 0.0)
                    try:
                        sim_val = float(sim)
                    except Exception:
                        sim_val = 0.0
                    try:
                        discount_val = float(discount)
                    except Exception:
                        discount_val = 0.0
                    override_parts.append(
                        f"{rule_name} 命中GOOD案例“{matched_good_case}”(sim={sim_val:.2f}, 抵扣={discount_val:.1f})"
                    )
                override_reason_suffix = "；GOOD案例覆盖：" + "；".join(override_parts)
            bad_case_reason_suffix = "；BAD案例触发：无"
            if bad_case_hits:
                bad_parts: List[str] = []
                for item in bad_case_hits:
                    rule_name = str(item.get("rule_name", "") or "")
                    matched_bad_case = str(item.get("matched_bad_case", "") or "")
                    sim = item.get("bad_case_score", 0.0)
                    bonus = item.get("bonus", 0.0)
                    try:
                        sim_val = float(sim)
                    except Exception:
                        sim_val = 0.0
                    try:
                        bonus_val = float(bonus)
                    except Exception:
                        bonus_val = 0.0
                    bad_parts.append(
                        f"{rule_name} 命中BAD案例“{matched_bad_case}”(sim={sim_val:.2f}, 加分={bonus_val:.1f})"
                    )
                bad_case_reason_suffix = "；BAD案例触发：" + "；".join(bad_parts)
            calibration_reason_suffix = "；易错说明校准：无"
            if calibration_hits:
                cal_parts: List[str] = []
                for item in calibration_hits:
                    rname = str(item.get("rule_name", "") or "")
                    ctype = str(item.get("type", "") or "")
                    terms = item.get("matched_terms", []) or []
                    delta = item.get("delta", 0.0)
                    ctype_text = "易误判校准" if ctype == "false_positive" else "易漏判校准"
                    cal_parts.append(f"{rname} {ctype_text} 命中{terms}({delta:+.1f})")
                calibration_reason_suffix = "；易错说明校准：" + "；".join(cal_parts)
            if not good_case_overrides:
                override_reason_suffix = "；GOOD案例覆盖：无"
            if summary_reason:
                reason_text = f"整体风险分 {risk_score:.1f}，决策 {final_decision}。{summary_reason}{bad_case_reason_suffix}{override_reason_suffix}{calibration_reason_suffix}"
            else:
                reason_text = f"整体风险分 {risk_score:.1f}，决策 {final_decision}。模型未给出详细说明{bad_case_reason_suffix}{override_reason_suffix}{calibration_reason_suffix}"

            e01_name = self._get_rule_name_by_id(1)
            should_rerun_direct_promise = (
                not _is_direct_promise_retry
                and bool(e01_name)
                and e01_name in event_scores
            )
            if should_rerun_direct_promise:
                return self.predict(
                    text,
                    product_type=product_type,
                    _is_direct_promise_retry=True,
                )

            # 8. 将 raw_response 规范化为三行文本输出，方便外部系统直接使用
            line_violation = f"是否违规：{'是' if violation else '否'}"
            line_events = f"触发事件：{triggered_event_str if triggered_event_str else '无'}"
            line_reason = f"理由：{reason_text}"
            raw_response = "\n".join([line_violation, line_events, line_reason])

            return {
                "raw_response": raw_response,
                "violation": violation,
                "triggered_event": triggered_event_str if violation else ("无" if triggered_event_str == "无" else triggered_event_str),
                "reason": reason_text,
                "event_reasons": formatted_event_reasons,
                # 新增评分相关字段，供上层使用
                "risk_score": risk_score,
                "decision": final_decision,
                "confidence": confidence,
                "risk_factors": risk_factors,
                "protective_factors": protective_factors,
                "bad_case_hits": bad_case_hits,
                "good_case_overrides": good_case_overrides,
                "calibration_hits": calibration_hits,
                "_debug": {
                    "violation_threshold": violation_threshold,
                    "review_threshold": review_threshold,
                    "original_violation": original_violation,
                    "original_triggered_event": original_triggered_event_str,
                    "original_event_reasons": original_event_reasons,
                    "product_type": product_type,
                    "normalized_product_type": pt,
                } if product_type is not None else None,
            }

        except Exception as e:
            return {
                "raw_response": f"系统错误: {str(e)}",
                "violation": False,
                "triggered_event": "系统错误",
                "reason": str(e),
                "event_reasons": {},
                "risk_score": 0.0,
                "decision": "compliant",
                "confidence": 0.0,
                "risk_factors": [],
                "protective_factors": [],
            }


    def debug_retrieval(self, text: str) -> Dict[str, Any]:
        """调试检索过程：候选规则 ID、最终送入的完整规则列表。"""
        candidate_ids = self._get_candidate_rule_ids(text)
        full_rule_docs = self._retrieve_rules_full(text)
        keyword_matches = self._keyword_match_rules(text)
        return {
            "input": text,
            "candidate_rule_ids": candidate_ids,
            "candidate_rule_names": [self._get_rule_name_by_id(rid) for rid in candidate_ids],
            "keyword_matched_rules": [
                {
                    "rule_id": rule_id,
                    "rule_name": self._get_rule_name_by_id(rule_id),
                    "score": score,
                    "matched_keywords": matched_keywords
                }
                for rule_id, score, matched_keywords in keyword_matches[:5]
            ],
            "final_full_rules_ordered": [doc.metadata.get("rule_name", "未知") for doc in full_rule_docs],
            "final_count": len(full_rule_docs),
            "max_rules": self._max_rules,
        }


