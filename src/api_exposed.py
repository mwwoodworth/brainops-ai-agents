from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.graph.product_agent import app as product_agent_graph
from langchain_core.messages import HumanMessage

app = FastAPI()

class ProductRequest(BaseModel):
    concept: str

@app.post("/agents/product/run")
async def run_product_agent(request: ProductRequest):
    """
    Run the LangGraph Product Agent
    """
    try:
        # Invoke the graph
        result = product_agent_graph.invoke({
            "messages": [HumanMessage(content=request.concept)]
        })
        
        # Extract the final message
        last_message = result["messages"][-1].content
        return {
            "status": "success",
            "result": last_message,
            "trace": [m.content for m in result["messages"]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}
