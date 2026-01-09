from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

# Define State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_step: str
    context: dict

# Nodes
def research_node(state: AgentState):
    print("--- RESEARCHING ---")
    # Call Gemini/Perplexity logic here
    return {"messages": [AIMessage(content="Market Analysis Complete")]}

def coding_node(state: AgentState):
    print("--- CODING ---")
    # Call Codex logic here
    return {"messages": [AIMessage(content="Code Generated")]}

def qa_node(state: AgentState):
    print("--- QA ---")
    # Call Playwright logic here
    return {"messages": [AIMessage(content="QA Passed")]}

# Supervisor / Router
def supervisor(state: AgentState):
    # Logic to decide next step
    last_message = state['messages'][-1].content
    if "Analysis Complete" in last_message:
        return "coding"
    if "Code Generated" in last_message:
        return "qa"
    return END

# Graph Construction
workflow = StateGraph(AgentState)

workflow.add_node("research", research_node)
workflow.add_node("coding", coding_node)
workflow.add_node("qa", qa_node)

workflow.set_entry_point("research")

workflow.add_conditional_edges(
    "research",
    supervisor,
    {"coding": "coding", END: END}
)
workflow.add_conditional_edges(
    "coding",
    supervisor,
    {"qa": "qa", END: END}
)
workflow.add_edge("qa", END)

app = workflow.compile()

if __name__ == "__main__":
    print("Running BrainOps LangGraph Agent...")
    result = app.invoke({"messages": [HumanMessage(content="Build a CRM")]})
    print(result)
