from typing import List, TypedDict, Annotated, Optional
import operator

class ClassificationResult(TypedDict):
    id: int
    AspectTerms: Optional[str]
    OpinionTerms: Optional[str]
    Categories: str
    Polarities: str
    Reasoning: str

class AgentState(TypedDict):
    # 输入的评论
    review: str
    # 评论 ID (如果提供)
    review_id: Optional[int]
    # RAG 检索到的相似示例
    examples: List[dict]
    # 当前分类结果
    results: List[ClassificationResult]
    # 评估反馈
    feedback: Optional[str]
    # 下一个要执行的节点 (用于 router)
    next_step: Optional[str]
    # 是否需要重新分类
    retry_count: int
    # 思考过程日志
    thought_process: Annotated[List[str], operator.add]
