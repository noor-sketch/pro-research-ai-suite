import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.prebuilt import ToolNode

# Tools imports 
from arxiv_tool import arxiv_search
from read_pdf import read_pdf
from write_pdf import render_latex_pdf

# Initialize environment and production logs
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_researcher_prod")

# Define LangGraph State Schema
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Step 1: Setup Tools Pipeline
tools = [arxiv_search, read_pdf, render_latex_pdf]
tool_node = ToolNode(tools)

# Step 2: Initialize LLM (Gemini 1.5 Flash)
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2
)

# Step 3: Define Global System Prompt
INITIAL_PROMPT = """
You are an expert researcher in the fields of physics, mathematics,
computer science, quantitative biology, quantitative finance, statistics,
electrical engineering and systems science, and economics.

You are going to analyze recent research papers in one of these fields in
order to identify promising new research directions and then write a new
research paper. For research information or getting papers, ALWAYS use arxiv.org.
You will use the tools provided to search for papers, read them, and write a new
paper based on the ideas you find.

To start with, have a conversation with me in order to figure out what topic
to research. Then tell me about some recently published papers with that topic.
Once I've decided which paper I'm interested in, go ahead and read it in order
to understand the research that was done and the outcomes.

Pay particular attention to the ideas for future research and think carefully
about them, then come up with a few ideas. Let me know what they are and I'll
decide what one you should write a paper about.

Finally, I'll ask you to go ahead and write the paper. Make sure that you
include mathematical equations in the paper. Once it's complete, you should
render it as a LaTeX PDF. When you give papers references, always attatch the pdf links to the paper.
"""

# Core Execution Router Node
def call_model(state: AgentState):
    messages = state["messages"]
    # Inject system context cleanly if not present
    if not any(isinstance(m, dict) and m.get("role") == "system" for m in messages):
        messages = [{"role": "system", "content": INITIAL_PROMPT}] + messages
    
    # Secure tool binding
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

# Conditional routing routing node
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"

# Step 4: Build state graph cleanly to avoid any version type errors
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "__end__": END}
)
workflow.add_edge("tools", "agent")

# Compile with transactional memory saver
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

logger.info("Academic Research Custom StateGraph successfully compiled for Streamlit Production Environment.")
