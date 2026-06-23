import logging
import re
import time
from io import BytesIO
from typing import Any
from uuid import uuid4

import requests
import streamlit as st
from fpdf import FPDF
from langchain_core.messages import AIMessage
from pypdf import PdfReader

from ai_researcher import INITIAL_PROMPT, build_config, graph
from arxiv_tool import clear_arxiv_cache, search_arxiv_papers


logging.basicConfig(level=logging.INFO)

MAX_DOCUMENT_CHARS = 7000
SIDEBAR_HISTORY_LIMIT = 8
MAX_RECENT_MESSAGES = 6
MAX_USER_MESSAGE_CHARS = 2200
MAX_ASSISTANT_MESSAGE_CHARS = 1400
MAX_MODEL_INPUT_CHARS = 11000
ARXIV_RESULTS_TTL_SECONDS = 1800


def initialize_session_state() -> None:
    defaults = {
        "chat_history": [],
        "conversation_messages": [],
        "last_response": "",
        "last_response_pdf": b"",
        "last_response_word_count": 0,
        "last_response_citations": 0,
        "last_uploaded_document": None,
        "latest_topic": "",
        "latest_topic_candidate": "",
        "latest_arxiv_results": [],
        "selected_arxiv_paper_index": 0,
        "session_thread_id": uuid4().hex,
        "request_counter": 0,
        "uploader_nonce": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session_state() -> None:
    st.session_state.chat_history = []
    st.session_state.conversation_messages = []
    st.session_state.last_response = ""
    st.session_state.last_response_pdf = b""
    st.session_state.last_response_word_count = 0
    st.session_state.last_response_citations = 0
    st.session_state.last_uploaded_document = None
    st.session_state.latest_topic = ""
    st.session_state.latest_topic_candidate = ""
    st.session_state.latest_arxiv_results = []
    st.session_state.selected_arxiv_paper_index = 0
    st.session_state.session_thread_id = uuid4().hex
    st.session_state.request_counter = 0
    st.session_state.uploader_nonce += 1
    st.cache_data.clear()
    st.cache_resource.clear()
    clear_arxiv_cache()


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def clean_preview(text: str, limit: int = 90) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."


def trim_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def export_pdf(title: str, text: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_fill_color(19, 38, 62)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 14, title.encode("latin-1", "ignore").decode("latin-1"), ln=True, fill=True)

    pdf.ln(8)
    pdf.set_text_color(60, 60, 60)
    pdf.set_font("Helvetica", size=11)
    cleaned_text = text.encode("latin-1", "ignore").decode("latin-1")
    pdf.multi_cell(0, 7, cleaned_text)
    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, str):
        return pdf_bytes.encode("latin-1", "ignore")
    return bytes(pdf_bytes)


def extract_pdf_text(uploaded_file: Any) -> tuple[str, int]:
    pdf_bytes = uploaded_file.getvalue()
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("The uploaded file does not appear to be a valid PDF.")
    reader = PdfReader(BytesIO(pdf_bytes))
    pages: list[str] = []

    for page in reader.pages:
        content = page.extract_text() or ""
        if content.strip():
            pages.append(content.strip())

    return "\n\n".join(pages), len(reader.pages)


def build_document_prompt(file_name: str, extracted_text: str, page_count: int) -> str:
    truncated_text = extracted_text[:MAX_DOCUMENT_CHARS]
    truncation_note = ""
    if len(extracted_text) > len(truncated_text):
        truncation_note = (
            f"\nNote: the document was truncated to the first {MAX_DOCUMENT_CHARS} characters "
            "to keep the summary request efficient.\n"
        )

    return (
        f'I uploaded a local PDF named "{file_name}".\n'
        f"Page count: {page_count}\n"
        f"Extracted word count: {count_words(extracted_text)}\n"
        "Use the uploaded document as the primary source.\n"
        "Do not use external search or tools unless I explicitly ask for outside research.\n"
        "Please provide:\n"
        "1. A synthesized executive summary.\n"
        "2. The main findings or arguments.\n"
        "3. Important evidence, methods, or technical points.\n"
        "4. Research gaps, limitations, or next steps.\n"
        "5. A short takeaway section.\n"
        f"{truncation_note}\n"
        "Document content:\n"
        f"{truncated_text}"
    )


def build_fallback_summary(document: dict[str, Any]) -> str:
    extracted_text = document["text"]
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(extracted_text.split()))
    selected_sentences: list[str] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if 50 <= len(sentence) <= 260:
            selected_sentences.append(sentence)
        if len(selected_sentences) == 4:
            break

    if not selected_sentences and extracted_text.strip():
        selected_sentences.append(clean_preview(extracted_text, 260))

    summary_lines = [
        f"Summary for {document['name']}",
        "",
        "Executive summary:",
        " ".join(selected_sentences[:2]) or "The document text was extracted, but only limited structure was available for summarization.",
        "",
        "Key points:",
    ]

    for index, sentence in enumerate(selected_sentences, start=1):
        summary_lines.append(f"{index}. {sentence}")

    summary_lines.extend(
        [
            "",
            "Research gaps and next steps:",
            "1. Review the full document for sections with figures, tables, or scanned pages that text extraction may have missed.",
            "2. Validate the main claims against cited sources if you plan to reuse the summary in a formal report.",
            "3. Ask a follow-up question in the chat for a more targeted synthesis, critique, or comparison.",
        ]
    )

    return "\n".join(summary_lines)


@st.cache_data(show_spinner=False, ttl=ARXIV_RESULTS_TTL_SECONDS)
def fetch_topic_papers(topic: str, max_results: int = 5) -> dict[str, Any]:
    """Fetch arXiv papers and return full result dict with entries and possible error."""
    result = search_arxiv_papers(topic, max_results=max_results)
    return result if isinstance(result, dict) else {"entries": [], "error": "Invalid response format"}


@st.cache_data(show_spinner=False)
def fetch_arxiv_pdf_bytes(pdf_url: str) -> bytes:
    """Fetch PDF with retries for transient errors only."""
    max_retries = 2
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = requests.get(pdf_url, timeout=60)
            response.raise_for_status()
            return response.content
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') else None
            # Don't retry on client errors (4xx) - the resource doesn't exist
            if status_code and 400 <= status_code < 500:
                logging.warning(f"PDF not available (HTTP {status_code}): {pdf_url}")
                raise Exception(f"PDF not available (HTTP {status_code})")
            # Retry on server errors (5xx)
            if attempt < max_retries - 1:
                logging.warning(f"PDF download server error (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(retry_delay)
            else:
                raise Exception(f"PDF download failed with HTTP {status_code}")
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                logging.warning(f"PDF download timeout (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(retry_delay)
            else:
                raise Exception("PDF download timed out after retries")
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                logging.warning(f"PDF download connection error (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(retry_delay)
            else:
                raise Exception("Unable to download PDF - connection failed")
        except Exception as e:
            logging.error(f"PDF download error: {e}")
            raise


def queue_user_message(display_content: str, backend_content: str | None = None) -> None:
    st.session_state.chat_history.append({"role": "user", "content": display_content})
    st.session_state.conversation_messages.append(
        {"role": "user", "content": backend_content or display_content}
    )


def store_assistant_response(response_text: str) -> None:
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
    st.session_state.conversation_messages.append({"role": "assistant", "content": response_text})
    st.session_state.last_response = response_text
    st.session_state.last_response_word_count = count_words(response_text)
    st.session_state.last_response_citations = len(re.findall(r"\[\d+\]", response_text))
    st.session_state.last_response_pdf = export_pdf("Research Summary Report", response_text)


def build_chat_input_messages(is_document_request: bool) -> list[dict[str, str]]:
    if is_document_request:
        latest_user_message = next(
            (
                message["content"]
                for message in reversed(st.session_state.conversation_messages)
                if message["role"] == "user"
            ),
            "",
        )
        return [
            {"role": "system", "content": INITIAL_PROMPT},
            {"role": "user", "content": trim_text(latest_user_message, MAX_DOCUMENT_CHARS)},
        ]

    recent_messages = st.session_state.conversation_messages[-MAX_RECENT_MESSAGES:]
    trimmed_messages: list[dict[str, str]] = []
    total_chars = len(INITIAL_PROMPT)

    for message in reversed(recent_messages):
        role = message["role"]
        limit = MAX_USER_MESSAGE_CHARS if role == "user" else MAX_ASSISTANT_MESSAGE_CHARS
        trimmed_content = trim_text(message["content"], limit)
        projected_total = total_chars + len(trimmed_content)
        if projected_total > MAX_MODEL_INPUT_CHARS and trimmed_messages:
            continue
        if projected_total > MAX_MODEL_INPUT_CHARS:
            remaining = max(MAX_MODEL_INPUT_CHARS - total_chars, 500)
            trimmed_content = trim_text(trimmed_content, remaining)
        trimmed_messages.append({"role": role, "content": trimmed_content})
        total_chars += len(trimmed_content)

    trimmed_messages.reverse()
    return [{"role": "system", "content": INITIAL_PROMPT}] + trimmed_messages


def clear_topic_papers(keep_candidate: bool = True) -> None:
    st.session_state.latest_topic = ""
    st.session_state.latest_arxiv_results = []
    st.session_state.selected_arxiv_paper_index = 0
    if not keep_candidate:
        st.session_state.latest_topic_candidate = ""


def update_topic_papers(topic: str) -> None:
    try:
        papers_result = fetch_topic_papers(topic)
        
        if isinstance(papers_result, dict):
            if "error" in papers_result and papers_result.get("error"):
                error_message = papers_result["error"]
                logging.warning(f"arXiv search returned error for '{topic}': {error_message}")
                clear_topic_papers()
                st.warning(
                    f"⚠️ Couldn't search arXiv: {error_message}\n\n"
                    f"Try again in a moment, or continue chatting without paper references."
                )
                return
            papers = papers_result.get("entries", [])
            notice = papers_result.get("notice")
        else:
            papers = papers_result if isinstance(papers_result, list) else []
            notice = None
    except Exception as exc:
        logging.exception("arXiv topic lookup failed: %s", exc)
        clear_topic_papers()
        st.warning(
            f"⚠️ Error searching arXiv: {str(exc)[:100]}\n\n"
            f"You can still chat and upload documents. Paper search will try again next time."
        )
        return

    st.session_state.latest_topic = topic
    st.session_state.latest_topic_candidate = topic
    st.session_state.latest_arxiv_results = papers
    st.session_state.selected_arxiv_paper_index = 0

    if notice:
        st.info(notice)

    if not papers:
        st.info(f"📄 No papers found for '{topic}'. Try a different search term.")


def run_research_pipeline(is_document_request: bool) -> None:
    full_response = ""
    request_config = build_config(
        f"{st.session_state.session_thread_id}-{st.session_state.request_counter}"
    )
    st.session_state.request_counter += 1

    chat_input = {"messages": build_chat_input_messages(is_document_request)}

    with st.status("Synthesizing response...", expanded=True) as status:
        st.write("Reviewing the available context and preparing the response.")
        container = st.chat_message("assistant")
        response_placeholder = container.empty()

        try:
            for state in graph.stream(chat_input, request_config, stream_mode="values"):
                messages = state.get("messages", [])
                if not messages:
                    continue

                message = messages[-1]
                if isinstance(message, AIMessage) and message.content:
                    full_response = (
                        message.content
                        if isinstance(message.content, str)
                        else str(message.content)
                    )
                    response_placeholder.markdown(full_response)

            if not full_response:
                raise ValueError("The model finished without returning text.")

            status.update(label="Response ready", state="complete", expanded=False)
        except Exception as exc:
            logging.exception("Research pipeline failed: %s", exc)
            error_text = str(exc)

            if is_document_request and st.session_state.last_uploaded_document:
                full_response = build_fallback_summary(st.session_state.last_uploaded_document)
                response_placeholder.markdown(full_response)
                status.update(label="Fallback summary generated", state="complete", expanded=False)
                st.warning(
                    "The live model request failed, so a local fallback summary was generated from the extracted document text."
                )
            elif "Request too large for model" in error_text or "rate_limit_exceeded" in error_text:
                st.warning(
                    "The request was larger than the model limit, so only the most recent context was used for an automatic retry."
                )
                trimmed_messages = build_chat_input_messages(is_document_request)
                minimal_messages = [trimmed_messages[0], *trimmed_messages[1:][-2:]]
                retry_input = {"messages": minimal_messages}
                full_response = ""
                try:
                    for state in graph.stream(retry_input, request_config, stream_mode="values"):
                        messages = state.get("messages", [])
                        if not messages:
                            continue
                        message = messages[-1]
                        if isinstance(message, AIMessage) and message.content:
                            full_response = (
                                message.content
                                if isinstance(message.content, str)
                                else str(message.content)
                            )
                            response_placeholder.markdown(full_response)
                    if not full_response:
                        raise ValueError("Retry finished without returning text.")
                    status.update(label="Response ready", state="complete", expanded=False)
                except Exception as retry_exc:
                    logging.exception("Trimmed retry failed: %s", retry_exc)
                    status.update(label="Request failed", state="error", expanded=True)
                    st.error(
                        "The request was too large for the current model window. Please shorten the prompt or clear old context and try again."
                    )
                    return
            else:
                status.update(label="Request failed", state="error", expanded=True)
                st.error(f"Execution error: {exc}")
                return

    store_assistant_response(full_response)


def render_sidebar_history() -> None:
    user_prompts = [
        message["content"]
        for message in st.session_state.chat_history
        if message["role"] == "user"
    ]

    st.subheader("Prompt History")
    if not user_prompts:
        st.caption("No user prompts yet.")
        return

    for prompt in reversed(user_prompts[-SIDEBAR_HISTORY_LIMIT:]):
        st.caption(clean_preview(prompt))


def render_report_panel() -> None:
    if not st.session_state.last_response and not st.session_state.latest_arxiv_results:
        return

    st.divider()
    st.subheader("Latest Summary")

    metric_1, metric_2, metric_3 = st.columns(3)
    with metric_1:
        st.metric("Summary words", st.session_state.last_response_word_count)
    with metric_2:
        document_words = (
            st.session_state.last_uploaded_document["word_count"]
            if st.session_state.last_uploaded_document
            else len(
                [
                    message
                    for message in st.session_state.chat_history
                    if message["role"] == "user"
                ]
            )
        )
        label = "Document words" if st.session_state.last_uploaded_document else "Prompt count"
        st.metric(label, document_words)
    with metric_3:
        st.metric("Bracketed citations", st.session_state.last_response_citations)

    if st.session_state.last_response_pdf:
        st.download_button(
            label="Download Summary PDF",
            data=st.session_state.last_response_pdf,
            file_name="research_summary_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    topic_candidate = st.session_state.latest_topic_candidate.strip()
    if topic_candidate:
        st.caption(
            "Paper lookup is manual now, so regular chat stays responsive and avoids arXiv rate limits."
        )
        if st.button("Find related arXiv papers", use_container_width=True):
            update_topic_papers(topic_candidate)

    if st.session_state.latest_arxiv_results:
        st.divider()
        st.subheader("Original arXiv Papers")
        if st.session_state.latest_topic:
            st.caption(f"Latest papers for: {st.session_state.latest_topic}")

        paper_options = [
            f"{index + 1}. {(paper.get('title') or 'Untitled paper')}"
            for index, paper in enumerate(st.session_state.latest_arxiv_results)
        ]
        selected_label = st.selectbox(
            "Choose a paper to download",
            paper_options,
            index=min(
                st.session_state.selected_arxiv_paper_index,
                max(len(paper_options) - 1, 0),
            ),
        )
        selected_index = paper_options.index(selected_label)
        st.session_state.selected_arxiv_paper_index = selected_index
        paper = st.session_state.latest_arxiv_results[selected_index]
        title = paper.get("title") or f"Paper {selected_index + 1}"
        authors = ", ".join(paper.get("authors", [])[:4]) or "Unknown authors"
        published = paper.get("published") or "Unknown date"
        summary = clean_preview(paper.get("summary", ""), limit=500)
        pdf_url = paper.get("pdf")
        abstract_url = paper.get("abstract_url")

        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(f"Authors: {authors}")
            st.caption(f"Published: {published}")
            if summary:
                st.write(summary)
            if abstract_url:
                st.markdown(f"[Open abstract page]({abstract_url})")

            if pdf_url:
                col_download, col_link = st.columns([2, 1])
                with col_download:
                    try:
                        pdf_bytes = fetch_arxiv_pdf_bytes(pdf_url)
                        file_stub = re.sub(r"[^a-zA-Z0-9_-]+", "_", title).strip("_") or f"paper_{selected_index + 1}"
                        st.download_button(
                            label="Download Original arXiv PDF",
                            data=pdf_bytes,
                            file_name=f"{file_stub}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key=f"download_arxiv_pdf_{selected_index}_{paper.get('id', selected_index)}",
                        )
                    except Exception as exc:
                        st.warning(f"Could not download the PDF: {exc}")
                with col_link:
                    st.link_button(
                        "Open PDF",
                        pdf_url,
                        use_container_width=True,
                    )


st.set_page_config(
    page_title="Pro Research Suite",
    page_icon=":material/auto_stories:",
    layout="wide",
)
initialize_session_state()

st.markdown(
    """
    <style>
    .stApp {
        --surface-bg: rgba(255, 255, 255, 0.9);
        --surface-border: rgba(148, 163, 184, 0.25);
        --surface-text: #0f172a;
        --muted-text: #334155;
        --sidebar-bg: linear-gradient(180deg, #334155 0%, #1e293b 100%);
        --sidebar-text: #f8fafc;
        --sidebar-panel-bg: rgba(255, 255, 255, 0.08);
        --sidebar-panel-border: rgba(255, 255, 255, 0.14);
        background:
            radial-gradient(circle at top left, rgba(29, 78, 216, 0.10), transparent 32%),
            linear-gradient(180deg, #f7fafc 0%, #eef4fb 100%);
    }
    @media (prefers-color-scheme: dark) {
        .stApp {
            --surface-bg: rgba(248, 250, 252, 0.94);
            --surface-border: rgba(148, 163, 184, 0.32);
            --surface-text: #020617;
            --muted-text: #1e293b;
            --sidebar-bg: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
            --sidebar-text: #f8fafc;
            --sidebar-panel-bg: rgba(255, 255, 255, 0.08);
            --sidebar-panel-border: rgba(255, 255, 255, 0.14);
        }
    }
    .hero-card {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: #ffffff;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.16);
        margin-bottom: 1rem;
    }
    .hero-card h1 {
        margin: 0;
        font-size: 2.5rem;
        color: #ffffff !important;
    }
    .hero-card p {
        margin: 0.45rem 0 0;
        color: #ffffff !important;
    }
    div[data-testid="stHeadingWithActionElements"] h1,
    div[data-testid="stHeadingWithActionElements"] h2,
    div[data-testid="stHeadingWithActionElements"] h3 {
        color: var(--surface-text) !important;
    }
    section[data-testid="stSidebar"] {
        background: var(--sidebar-bg);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: var(--sidebar-text) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] p {
        color: var(--sidebar-text) !important;
        opacity: 1 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"],
    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-testid="stTextInputRootElement"],
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] input {
        color: var(--sidebar-text) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] section,
    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-testid="stTextInputRootElement"],
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] input {
        background: var(--sidebar-panel-bg) !important;
        border-color: var(--sidebar-panel-border) !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(90deg, #0f172a 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(90deg, #172554 0%, #2563eb 100%) !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stInfo"],
    section[data-testid="stSidebar"] div[data-testid="stSuccess"],
    section[data-testid="stSidebar"] div[data-testid="stWarning"],
    section[data-testid="stSidebar"] div[data-testid="stError"] {
        background: var(--sidebar-panel-bg);
        border: 1px solid var(--sidebar-panel-border);
    }
    section[data-testid="stSidebar"] div[data-testid="stInfo"] *,
    section[data-testid="stSidebar"] div[data-testid="stSuccess"] *,
    section[data-testid="stSidebar"] div[data-testid="stWarning"] *,
    section[data-testid="stSidebar"] div[data-testid="stError"] * {
        color: var(--sidebar-text) !important;
    }
    .stChatMessage {
        border-radius: 16px;
        border: 1px solid var(--surface-border);
        background: var(--surface-bg);
        color: var(--surface-text);
    }
    .stChatMessage p,
    .stChatMessage li,
    .stChatMessage span,
    .stChatMessage strong,
    .stChatMessage label,
    .stChatMessage code,
    .stChatMessage pre,
    .stChatMessage h1,
    .stChatMessage h2,
    .stChatMessage h3,
    .stChatMessage h4,
    .stChatMessage h5,
    .stChatMessage h6 {
        color: var(--surface-text) !important;
    }
    .stMarkdown,
    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown span,
    .stMarkdown strong,
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4,
    .stMarkdown h5,
    .stMarkdown h6 {
        color: var(--surface-text) !important;
    }
    div[data-testid="stMetric"] {
        background: var(--surface-bg);
        border: 1px solid var(--surface-border);
        border-radius: 16px;
        padding: 0.8rem;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] p,
    div[data-testid="stMetricValue"] {
        color: var(--surface-text) !important;
    }
    div[data-testid="stCaptionContainer"] p,
    div[data-testid="stInfo"] *,
    div[data-testid="stSuccess"] *,
    div[data-testid="stWarning"] *,
    div[data-testid="stError"] * {
        color: var(--muted-text) !important;
    }
    div[data-testid="stDownloadButton"] > button {
        width: 100%;
        border-radius: 999px;
        border: none;
        padding: 0.78rem 1rem;
        font-weight: 700;
        color: #ffffff;
        background: linear-gradient(90deg, #0f172a 0%, #1d4ed8 100%);
        box-shadow: 0 12px 24px rgba(29, 78, 216, 0.25);
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(90deg, #172554 0%, #2563eb 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <h1>Pro Research AI Suite</h1>
        <p>Upload a paper, ask follow-up questions, and export the latest synthesized summary as a PDF.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

pending_document_request = False

with st.sidebar:
    st.header("Research Workspace")
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key=f"document_uploader_{st.session_state.uploader_nonce}",
    )

    if uploaded_file is not None:
        st.caption(f"Selected file: {uploaded_file.name}")

    analyze_clicked = st.button(
        "Analyze Uploaded Document",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

    if analyze_clicked and uploaded_file is not None:
        with st.spinner("Extracting text from the uploaded document..."):
            try:
                extracted_text, page_count = extract_pdf_text(uploaded_file)
            except Exception as exc:
                st.error(f"Unable to read the PDF: {exc}")
            else:
                if not extracted_text.strip():
                    st.error(
                        "No readable text was extracted from the PDF. It may be image-only or scanned."
                    )
                else:
                    word_count = count_words(extracted_text)
                    st.session_state.last_uploaded_document = {
                        "name": uploaded_file.name,
                        "page_count": page_count,
                        "word_count": word_count,
                        "text": extracted_text,
                    }
                    clear_topic_papers(keep_candidate=False)
                    queue_user_message(
                        f"Summarize the uploaded document: {uploaded_file.name}",
                        build_document_prompt(uploaded_file.name, extracted_text, page_count),
                    )
                    pending_document_request = True
                    st.success("Document queued for synthesis.")

    if st.session_state.last_uploaded_document:
        document = st.session_state.last_uploaded_document
        st.info(
            f"Last upload: {document['name']}\n\nPages: {document['page_count']}\n\nWords: {document['word_count']}"
        )

    st.divider()
    render_sidebar_history()
    st.divider()

    if st.button("Clear Cache", use_container_width=True):
        reset_session_state()
        st.rerun()

user_input = st.chat_input("Ask a research question or summarize the uploaded document...")
pending_user_request = False

if user_input:
    queue_user_message(user_input)
    st.session_state.latest_topic_candidate = user_input
    clear_topic_papers()
    pending_user_request = True

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if pending_document_request or pending_user_request:
    run_research_pipeline(is_document_request=pending_document_request)

render_report_panel()
# py -3.13 -m streamlit run frontend.py
