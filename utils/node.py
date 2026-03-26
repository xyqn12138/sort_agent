from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage
from .state import AgentState, ClassificationResult
from rag.ragservice import RAGService
from .tools import MySQLService
from .config_loader import config
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 初始化服务
model_name = config.get('model', {}).get('name', 'qwen-max')
llm = ChatTongyi(model=model_name)
rag_service = RAGService()
mysql_service = MySQLService()

# 调试开关
VERBOSE = config.get('debug', {}).get('verbose', False)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def llm_invoke_with_retry(messages):
    return llm.invoke(messages)

def retrieval_node(state: AgentState):
    """检索相似评论及对应标签"""
    review = state['review']
    thought = [f"Retrieval: Searching for reviews similar to '{review[:30]}...'"]
    
    # 1. RAG 检索相似评论
    similar_reviews = rag_service.search(review, k=3)
    thought.append(f"Retrieval: Found {len(similar_reviews)} similar reviews in VectorDB")
    
    examples = []
    for item in similar_reviews:
        # 2. 从 MySQL 获取对应标签
        labels = mysql_service.get_labels_by_id(item['id'])
        examples.append({
            "review": item['content'],
            "labels": labels
        })
    
    thought.append(f"Retrieval: Loaded labels from MySQL for all matched IDs")
    return {"examples": examples, "thought_process": thought}

def classification_node(state: AgentState):
    """执行分类逻辑"""
    review = state['review']
    review_id = state.get('review_id')
    examples = state['examples']
    feedback = state.get('feedback', '')
    retry = state.get('retry_count', 0)
    previous_results = state.get('results', [])
    
    thought = [f"Classification: Attempt {retry + 1}"]
    
    # 构造历史修正区块
    correction_block = ""
    if feedback:
        thought.append(f"Classification: Applying feedback for precision correction")
        # 避免展示空的上一轮结果误导模型
        prev_res_str = json.dumps(previous_results, ensure_ascii=False) if previous_results else "（上一轮未产出有效结果）"
        correction_block = f"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
### 【重要：上一轮分类失败，请务必吸取教训】
- 上一轮输出的结果：{prev_res_str}
- 报错信息：{feedback}
- 请逐字核对目标评论，确保 AspectTerms 和 OpinionTerms 与原句【字符级】匹配。
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

    # 构建 Few-shot Prompt
    example_str = ""
    for i, ex in enumerate(examples):
        example_str += f"Example {i+1}:\nReview: {ex['review']}\nLabels: {json.dumps(ex['labels'], ensure_ascii=False)}\n\n"
    
    cat_str = ",".join(["包装", "成分", "尺寸", "服务", "功效", "价格", "气味", "使用体验", "物流", "新鲜度", "真伪", "整体", "其他"])
    pol_str = ",".join(["正面", "中性", "负面"])

    prompt = f"""你是一个专业的评论分类专家。请分析目标评论并输出 JSON 列表。

### 核心规则（严格遵守，违者必究）：
1. **【零容忍】词源一致性**：AspectTerms 和 OpinionTerms 必须【100%完全原封不动】地取自原评论。
   - **严禁**：漏掉、增加、改写任何字符。
   - **重点**：对“的”、“了”、“得”、“太”等虚词极其敏感。例如原句是“好用的”，你不能写成“好用”；原句是“非常的好用”，你不能写成“非常好用”或“非常好的”。
   - **自检**：输出前请逐字对比原句。如果 `find(original_text, term) == -1`，说明你写错了。
2. **分类枚举约束**：Categories 必须在 [{cat_str}] 中；Polarities 必须在 [{pol_str}] 中。
3. **兜底策略**：若属性不明确，AspectTerms 填 "_"。
4. **输出格式**：只输出合法的 JSON 列表。

### 参考示例：
{example_str}

### 待处理目标评论：
{review}
{correction_block}

请直接输出 JSON，不要有任何废话：
[
  {{"AspectTerms": "...", "OpinionTerms": "...", "Categories": "...", "Polarities": "...", "Reasoning": "请先自检：我提取的词是否与原句字符完全一致？然后简述逻辑"}}
]
"""
    
    # 调试输出：打印发送给 LLM 的核心内容
    if VERBOSE:
        print("\n--- [DEBUG] Classification Prompt Start ---")
        print(prompt)
        print("--- [DEBUG] Classification Prompt End ---\n")

    response = llm_invoke_with_retry([SystemMessage(content="你是一个评论分类专家，只输出 JSON 列表"), HumanMessage(content=prompt)])
    
    # 强制更新重试计数
    new_retry_count = retry + 1
    
    try:
        content = response.content.strip()
        # 更强大的 JSON 提取逻辑
        if "[" in content and "]" in content:
            # 找到第一个 [ 和最后一个 ]
            start = content.find("[")
            end = content.rfind("]") + 1
            content = content[start:end]
        
        results = json.loads(content)
        
        # 注入 review_id
        for res in results:
            if review_id:
                res['id'] = review_id
            thought.append(f"Reasoning: {res.get('Reasoning', 'N/A')}")
        
        return {"results": results, "retry_count": new_retry_count, "thought_process": thought}
    except Exception as e:
        err_msg = f"Error: Failed to parse JSON: {str(e)}"
        # 即使报错也要返回 new_retry_count
        return {"results": [], "feedback": "JSON 解析失败，请严格按照 JSON 列表格式输出", "retry_count": new_retry_count, "thought_process": [err_msg]}

def evaluation_node(state: AgentState):
    """评估分类结果，提高召回率"""
    review = state['review']
    results = state['results']
    examples = state['examples']
    thought = [f"Evaluation: Checking {len(results)} extraction results"]
    
    if not results:
        return {"next_step": "classify", "feedback": "未能生成分类结果，请重新尝试。", "thought_process": ["Warning: Empty results, triggering retry"]}

    # 定义合法集合
    valid_categories = ["包装", "成分", "尺寸", "服务", "功效", "价格", "气味", "使用体验", "物流", "新鲜度", "真伪", "整体", "其他"]
    valid_polarities = ["正面", "中性", "负面"]
    cat_str = ",".join(valid_categories)
    pol_str = ",".join(valid_polarities)

    # 预检：程序化检查字段是否符合硬约束
    errors = []
    for i, res in enumerate(results):
        aspect = str(res.get("AspectTerms", "")).strip()
        opinion = str(res.get("OpinionTerms", "")).strip()
        category = str(res.get("Categories", "")).strip()
        polarity = str(res.get("Polarities", "")).strip()
        
        # 1. 来源检查 (字面量硬匹配)
        if aspect != "_" and aspect not in review:
            errors.append(f"Result {i+1}: AspectTerms '{aspect}' 不在原评论中。请检查是否漏掉了字符（如'的'、'了'、'得'）或改变了字序。")
        if opinion != "_" and opinion not in review:
            errors.append(f"Result {i+1}: OpinionTerms '{opinion}' 不在原评论中。请检查是否漏掉了字符（如'的'、'了'、'得'）或改变了字序。")
        if category not in valid_categories:
            errors.append(f"Result {i+1}: Category '{category}' is invalid. Allowed: {cat_str}")
        if polarity not in valid_polarities:
            errors.append(f"Result {i+1}: Polarity '{polarity}' is invalid. Allowed: {pol_str}")

    if errors:
        feedback = f"分类结果不满足硬性约束条件：\n" + "\n".join(errors)
        feedback += f"\n请务必从合法集合中选择分类。合法类别集合: {cat_str}；合法极性集合: {pol_str}。"
        thought.append(f"Error: Hard constraint validation failed with {len(errors)} errors")
        
        # 增加重试次数检查，防止硬性约束错误导致无限循环
        if state.get('retry_count', 0) >= 2:
            thought.append("Evaluation: Max retries reached even with hard errors, forced termination")
            return {"next_step": "end", "results": results, "feedback": None, "thought_process": thought}
            
        # 显式透传 results，确保状态不丢失
        return {"next_step": "classify", "results": results, "feedback": feedback, "thought_process": thought}
    
    thought.append("Evaluation: All field constraints passed pre-check")
    
    # 构造 Few-shot 示例字符串供评估器参考
    example_str = ""
    for i, ex in enumerate(examples):
        example_str += f"Example {i+1}:\nReview: {ex['review']}\nLabels: {json.dumps(ex['labels'], ensure_ascii=False)}\n\n"

    # 逻辑评估：交给 LLM 检查完整性和业务逻辑
    prompt = f"""你是一个严谨的评论分类评估专家。请根据以下标准评估分类结果：

### 评估标准：
1. **词源一致性**：AspectTerms 和 OpinionTerms 必须【完全原封不动】地取自原评论。
2. **分类枚举约束**：
   - Categories 必须属于：{cat_str}
   - Polarities 必须属于：{pol_str}
3. **完整性**：是否遗漏了评论中提到的任何重要属性或观点？

### 参考示例：
{example_str}

### 待处理目标评论：
{review}

### 当前分类结果：
{json.dumps(results, ensure_ascii=False, indent=2)}

### 输出要求：
- 如果分类结果完全符合要求且没有遗漏，请回复 'PASS'。
- 如果不符合，请简洁列出错误点，并给出具体的修改建议（必须符合上述约束条件）。
"""
    
    # 调试输出
    if VERBOSE:
        print("\n--- [DEBUG] Evaluation Prompt Start ---")
        print(prompt)
        print("--- [DEBUG] Evaluation Prompt End ---\n")

    response = llm_invoke_with_retry([SystemMessage(content="你是一个严格的质量评估员"), HumanMessage(content=prompt)])
    feedback = response.content
    
    if "PASS" in feedback.upper() or state['retry_count'] >= 2:
        thought.append("Evaluation: Final PASS or max retries reached")
        return {"next_step": "end", "results": results, "feedback": None, "thought_process": thought}
    else:
        thought.append(f"Evaluation: Failed logic check, returning feedback to classifier")
        return {"next_step": "classify", "results": results, "feedback": feedback, "thought_process": thought}
