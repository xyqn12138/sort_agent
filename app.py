import streamlit as st
import pandas as pd
from agent import build_graph
from utils.config_loader import config
import json

# 设置页面配置
st.set_page_config(
    page_title="评论分类 Agent 交互系统",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 评论分类 Agent 交互系统")
st.markdown("请输入一条评论，Agent 将对其进行细粒度的情感与属性分析。")

# 初始化 Graph
if "app" not in st.session_state:
    st.session_state.app = build_graph()

# 输入框
review_input = st.text_area("待分析评论", placeholder="例如：第一次购买，遮瑕，保湿效果非常好。下次还买。", height=100)

if st.button("开始分析", type="primary"):
    if not review_input.strip():
        st.warning("请输入评论内容后再进行分析。")
    else:
        # 初始化状态
        inputs = {
            "review": review_input,
            "retry_count": 0,
            "results": [],
            "examples": [],
            "thought_process": []
        }
        
        # 思考过程展示容器
        thought_container = st.empty()
        all_thoughts = []
        
        # 结果展示区域
        final_state = inputs
        
        # 运行 Agent 并流式展示思考过程
        with st.spinner("Agent 正在深度思考并分类中..."):
            for output in st.session_state.app.stream(inputs, config={"recursion_limit": 15}):
                for node_name, state_update in output.items():
                    # 合并状态
                    final_state.update(state_update)
                    
                    # 提取并展示思考过程
                    if "thought_process" in state_update:
                        new_thoughts = state_update["thought_process"]
                        for t in new_thoughts:
                            all_thoughts.append(f"**[{node_name}]** {t}")
                        
                        # 动态更新思考过程 (保持展开)
                        with thought_container.expander("🔍 思考过程 (分析中...)", expanded=True):
                            st.markdown("\n\n".join(all_thoughts))
            
            # 分析完成后，用一个折叠的 Expander 替换掉刚才展开的
            with thought_container.expander("✅ 思考过程 (已完成)", expanded=False):
                st.markdown("\n\n".join(all_thoughts))
            
            # 展示最终结果
            st.subheader("📊 分类结果")
            results = final_state.get("results", [])
            
            if results:
                # 转换结果为 DataFrame 展示
                display_data = []
                for res in results:
                    display_data.append({
                        "属性 (Aspect)": res.get("AspectTerms", "_"),
                        "观点 (Opinion)": res.get("OpinionTerms", "_"),
                        "类别 (Category)": res.get("Categories", "其他"),
                        "极性 (Polarity)": res.get("Polarities", "中性")
                    })
                
                df = pd.DataFrame(display_data)
                st.table(df)
                
                # 同时展示原始 JSON
                with st.expander("查看原始 JSON"):
                    st.json(results)
            else:
                st.error("未能提取到有效分类结果，请检查评论内容或稍后重试。")

# 侧边栏信息
st.sidebar.header("关于系统")
st.sidebar.info(
    """
    本系统基于 **LangGraph** 构建，采用 **Evaluator-Reflector** 模式。
    
    **核心特点：**
    1. **RAG 增强**：检索相似案例进行 Few-shot 学习。
    2. **自我评估**：内置评分节点，自动修正不符合约束的结果。
    3. **词源一致性**：严格确保 Aspect 和 Opinion 来自原句。
    """
)
st.sidebar.markdown("---")
st.sidebar.text(f"Model: {config.get('model', {}).get('name', 'N/A')}")
