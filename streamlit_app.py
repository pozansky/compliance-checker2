import os
import sys
from typing import Optional

import streamlit as st


def _ensure_project_on_syspath() -> None:
    """Ensure `src.*` imports work when run via streamlit."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)


@st.cache_resource
def _get_engine():
    _ensure_project_on_syspath()
    # Import here so Streamlit caching works cleanly
    from rag_engine import ComplianceRAGEngine

    return ComplianceRAGEngine()


def _normalize_product_type(pt_label: str) -> Optional[str]:
    if pt_label == "全部检测":
        return None
    return pt_label


def main() -> None:
    st.set_page_config(page_title="合规检测系统", page_icon="🔍", layout="wide")

    st.title("🔍 合规检测系统（统一使用 rag_engine.py）")
    st.caption("此页面只负责界面展示，所有规则与判断逻辑都在 `rag_engine.py` 维护。")

    with st.sidebar:
        st.header("配置")
        product_type_label = st.selectbox(
            "选择产品类型",
            ["全部检测", "1.0", "2.0", "3.0"],
            index=0,
            help="选择产品类型会影响部分规则的生效范围",
        )
        show_debug = st.checkbox("显示检索调试信息", value=False)
        st.markdown("---")
        st.markdown(
            "- **直接输入**：支持粘贴多行聊天内容\n"
            "- **上传文件**：支持 `.txt`（UTF-8）\n"
        )

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        input_method = st.radio("输入方式", ["直接输入", "上传文件"], horizontal=True)
        input_text = ""
        if input_method == "直接输入":
            input_text = st.text_area(
                "请输入聊天内容",
                height=220,
                placeholder="例如：你就用手机号短信验证就可以了……",
            )
        else:
            uploaded = st.file_uploader("上传文本文件（.txt）", type=["txt"])
            if uploaded is not None:
                try:
                    input_text = uploaded.read().decode("utf-8", errors="ignore")
                    st.success(f"已读取文件：{uploaded.name}（{len(input_text)} 字符）")
                except Exception as e:
                    st.error(f"读取文件失败：{e}")

        run_btn = st.button("开始检测", type="primary", use_container_width=True)

    with col2:
        st.subheader("检测结果")
        if run_btn:
            text = (input_text or "").strip()
            if not text:
                st.warning("请输入聊天内容后再检测。")
                return

            engine = _get_engine()
            pt = _normalize_product_type(product_type_label)

            with st.spinner("分析中..."):
                result = engine.predict(text, product_type=pt)

            violation = bool(result.get("violation"))
            triggered_event = str(result.get("triggered_event", "无") or "无")
            reason = str(result.get("reason", "") or "")
            raw_response = str(result.get("raw_response", "") or "")

            # 顶部状态条
            if violation:
                st.error("判定：违规", icon="⛔")
            else:
                st.success("判定：合规", icon="✅")

            # 关键信息
            st.markdown(f"**触发事件：** {triggered_event}")

            tabs = st.tabs(["理由", "原始输出", "调试"])

            with tabs[0]:
                # 用 text_area 强制完整展示（st.write 有时会折叠/渲染得不明显）
                if reason.strip():
                    st.text_area("理由", value=reason, height=220, disabled=True)
                else:
                    st.warning("本次未返回可展示的理由（reason 为空）。建议查看“原始输出”。")

            with tabs[1]:
                if raw_response.strip():
                    st.code(raw_response, language=None)
                else:
                    st.info("本次未返回 raw_response。")

            with tabs[2]:
                if show_debug:
                    debug = engine.debug_retrieval(text)
                    st.json(debug, expanded=False)
                else:
                    st.info("侧边栏勾选“显示检索调试信息”后，这里会展示检索详情。")


if __name__ == "__main__":
    main()

