from langgraph.graph import StateGraph, END
from utils.state import AgentState
from utils.node import retrieval_node, classification_node, evaluation_node
import json

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("classify", classification_node)
    workflow.add_node("evaluate", evaluation_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "classify")
    workflow.add_edge("classify", "evaluate")

    def router(state: AgentState):
        if state["next_step"] == "classify":
            return "classify"
        return END
    
    workflow.add_conditional_edges("evaluate", router, {"classify": "classify", END: END})
    return workflow.compile()

def run_agent(review_text: str, verbose: bool = False):
    app = build_graph()
    inputs = {
        "review": review_text,
        "retry_count": 0,
        "results": [],
        "examples": [],
        "thought_process": []
    }
    
    if verbose:
        print(f"--- Processing Review: {review_text} ---")
    
    last_thought_idx = 0
    final_state = inputs
    # 只运行一次 stream，从中收集所有更新
    for output in app.stream(inputs, config={"recursion_limit": 10}):
        for node_name, state_update in output.items():
            # 将节点输出合并到最终状态中
            final_state.update(state_update)
            if verbose and "thought_process" in state_update:
                new_thoughts = state_update["thought_process"]
                # 每一个节点返回的都是其自身的 new_thoughts，直接打印即可
                for thought in new_thoughts:
                    print(f"[Node: {node_name}] {thought}")
            
    if verbose:
        print("\n--- Final Results ---")
        print(json.dumps(final_state.get("results", []), indent=2, ensure_ascii=False))
    return final_state

if __name__ == "__main__":
    test_review = "6,很细腻的一款产品，贴合肤色，一下子提亮了肌肤，很好"
    run_agent(test_review, verbose=True)
