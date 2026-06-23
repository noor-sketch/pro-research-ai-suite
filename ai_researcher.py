import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
import logging

# Tools imports (Ensuring absolute sync with project infrastructure)
from arxiv_tool import arxiv_search
from read_pdf import read_pdf
from write_pdf import render_latex_pdf

# Initialize environment and production logs
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_researcher_prod")

# Step 1: Setup Core Academic Tools Pipeline
tools = [arxiv_search, read_pdf, render_latex_pdf]

# Step 2: Initialize Enterprise-Grade LLM (Gemini 1.5 Flash)
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Step 3: Define Global System Prompt for Complex Research Orchestration
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

# Step 4: Compile the ReAct Agent Graph for Streamlit Production Pipeline
graph = create_react_agent(model, tools=tools, state_modifier=INITIAL_PROMPT)

# Step 5: Strict Module Exports & Verification Logging
__all__ = ["graph", "INITIAL_PROMPT"]

logger.info("Academic Research ReAct Graph successfully compiled for Streamlit Cloud production pipeline.")
