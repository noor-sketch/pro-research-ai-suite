import logging
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from arxiv_tool import search_arxiv_papers
from read_pdf import read_pdf
from write_pdf import render_latex_pdf


load_dotenv()
logging.basicConfig(level=logging.INFO)


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def safe_arxiv_search(query: str) -> dict:
    """Search arXiv papers with shared throttling and caching safeguards."""
    return search_arxiv_papers(query)


tools = [safe_arxiv_search, read_pdf, render_latex_pdf]
tool_node = ToolNode(tools)

model_name = "llama-3.1-8b-instant"

base_model = ChatGroq(
    model_name=model_name,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
    max_retries=2,
)

tool_model = base_model.bind_tools(tools)

INITIAL_PROMPT = """
You are a World-Class Academic AI Researcher. You have four primary mandates:

1. Knowledge source priority:
   - If the user provides text from an uploaded document, prioritize that information.
   - Use safe_arxiv_search only when outside research is actually needed.

2. Tool-call discipline:
   - Use safe_arxiv_search only for recent external research questions.
   - Do not call tools for plain summarization of uploaded document content.
   - When calling safe_arxiv_search, pass a plain string query.

3. Response quality:
   - Produce clear, grounded academic summaries.
   - When summarizing a document, include an executive summary, key findings, methods or evidence, limitations, and next steps.
   - Use citations only when you genuinely have source support for them.

4. Output flexibility:
   - Support Urdu for discussion if requested, but default to clear academic English.
   - Use render_latex_pdf only when the user explicitly asks to generate a paper or PDF through the agent.
"""

DIRECT_RESPONSE_PROMPT = """
Answer directly without calling any tools.
Keep the response in clean markdown.
If uploaded document content is present, ground the summary in that content first.
"""


def _message_role(message) -> str | None:
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "type", None)


def _message_content(message) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _ensure_system_prompt(messages: list) -> list:
    if any(_message_role(message) == "system" for message in messages):
        return list(messages)
    return [{"role": "system", "content": INITIAL_PROMPT}, *messages]


def _latest_user_message(messages: list) -> str:
    for message in reversed(messages):
        if _message_role(message) == "user":
            return _message_content(message)
    return ""


def _is_uploaded_document_summary(messages: list) -> bool:
    latest_user = _latest_user_message(messages)
    return 'I uploaded a local PDF named "' in latest_user and "Document content:" in latest_user


def _invoke_direct_response(messages: list) -> AIMessage:
    direct_messages = _ensure_system_prompt(messages)
    first_message = direct_messages[0]
    combined_system_prompt = INITIAL_PROMPT + "\n\n" + DIRECT_RESPONSE_PROMPT

    if _message_role(first_message) == "system":
        direct_messages[0] = {"role": "system", "content": combined_system_prompt}
    else:
        direct_messages.insert(0, {"role": "system", "content": combined_system_prompt})

    response = base_model.invoke(direct_messages)
    content = response.content if isinstance(response.content, str) else str(response.content)
    return AIMessage(content=content)


def call_model(state: State):
    messages = _ensure_system_prompt(state["messages"])

    if _is_uploaded_document_summary(messages):
        response = _invoke_direct_response(messages)
        return {"messages": [response]}

    try:
        response = tool_model.invoke(messages)
        return {"messages": [response]}
    except Exception as exc:
        error_text = str(exc)
        if "tool_use_failed" in error_text or "Failed to call a function" in error_text:
            logging.warning("Tool call formatting failed. Retrying once without tools.")

            if "render_latex_pdf" in error_text:
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "### Document Generated\n"
                                "The research draft was prepared, but the PDF rendering tool did not complete in this run."
                            )
                        )
                    ]
                }

            response = _invoke_direct_response(messages)
            return {"messages": [response]}

        raise exc


def should_continue(state: State) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "__end__": END},
)
workflow.add_edge("tools", "agent")

checkpointer = MemorySaver()


def build_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


config = build_config("222222")
graph = workflow.compile(checkpointer=checkpointer)
