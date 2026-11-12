
import html
import os

import requests
import streamlit as st

try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except Exception:
    BACKEND_URL = os.getenv("BACKEND_URL", "https://my-doc-reg-project.onrender.com")

API_URL = BACKEND_URL.rstrip("/")

# Render free server can take 50-60 seconds to wake up
WAKE_TIMEOUT = 120


UPLOAD_TIMEOUT = 900   # scanned documents (OCR) can take several minutes
CHAT_TIMEOUT = 180

SUPPORTED_TYPES = [
    "pdf", "docx", "xlsx", "xlsm", "pptx",
    "txt", "md", "csv", "rtf",
    "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"
]

FILE_ICONS = {
    "pdf": "📕", "docx": "📘", "xlsx": "📗", "xlsm": "📗", "pptx": "📙",
    "txt": "📝", "md": "📝", "csv": "📊", "rtf": "📝",
    "png": "🖼️", "jpg": "🖼️", "jpeg": "🖼️", "webp": "🖼️", "bmp": "🖼️",
    "tif": "🖼️", "tiff": "🖼️"
}


def error_detail(response, default):
    try:
        return response.json().get("detail", default)
    except Exception:
        return default


def file_icon(name):
    return FILE_ICONS.get(name.rsplit(".", 1)[-1].lower(), "📄")


def source_card(source):
    name = html.escape(str(source.get("source", "Unknown")))
    where = html.escape(str(source.get("location") or f"Page {source.get('page', '?')}"))
    return f'<div class="source-card">{file_icon(str(source.get("source", "")))} {name} · {where}</div>'

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #080d1b 0%, #10172a 55%, #11152b 100%);
    color: #f1f5f9;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

section[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid #26334d;
}

section[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

.brand {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #94a3b8;
    font-size: 14px;
}

.hero {
    padding: 28px;
    border-radius: 20px;
    background: linear-gradient(120deg, #172554, #312e81, #4c1d95);
    border: 1px solid #45418d;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #dbeafe;
    margin-bottom: 0;
}

.status-card {
    padding: 14px;
    border-radius: 12px;
    background: #111c30;
    border: 1px solid #263650;
    margin-bottom: 15px;
}

.document-card {
    padding: 10px 12px;
    margin: 7px 0;
    border-radius: 10px;
    background: #131f33;
    border: 1px solid #263650;
    font-size: 12px;
    overflow-wrap: anywhere;
}

.source-card {
    padding: 10px 14px;
    margin: 6px 0;
    border-radius: 10px;
    background: #17243a;
    border-left: 3px solid #818cf8;
    color: #cbd5e1;
    font-size: 12px;
}

div[data-testid="stChatMessage"] {
    background: #121d30;
    border: 1px solid #273650;
    border-radius: 16px;
    padding: 14px;
    margin-bottom: 12px;
}

.stChatInput textarea {
    background: #111c30 !important;
    color: white !important;
}

.stButton button {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
}

.stButton button:hover {
    border: 1px solid #a5b4fc;
    color: white;
}

div[data-testid="stExpander"] {
    background: #111c30;
    border: 1px solid #263650;
    border-radius: 12px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown('<div class="brand">📚 DocuMind</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your AI Knowledge Assistant</p>', unsafe_allow_html=True)

    st.divider()

    st.markdown("### ⚙️ System Status")

    try:
        with st.spinner("Server is waking up, please wait (up to 1 minute)..."):
            health = requests.get(f"{API_URL}/health", timeout=WAKE_TIMEOUT)
        if health.status_code == 200:
            st.success("Backend Connected")
        else:
            st.error("Backend Error")
    except requests.exceptions.RequestException:
        st.error("Backend Disconnected")

    st.divider()

    st.markdown("### 📂 Your Documents")

    document_names = []
    selected_source = "All Documents"

    try:
        response = requests.get(f"{API_URL}/documents", timeout=WAKE_TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            document_names = data["documents"]
            st.metric("Total Documents", data["total_documents"])

            for document in document_names:
                st.markdown(
                    f'<div class="document-card">{file_icon(document)} {html.escape(document)}</div>',
                    unsafe_allow_html=True
                )

            selected_source = st.selectbox(
                "🔎 Filter answers by document",
                ["All Documents"] + document_names
            )

        else:
            st.warning("Could not load documents.")

    except requests.exceptions.RequestException:
        st.error("Unable to connect to document service.")

    # ---------------- PDF UPLOAD ----------------

    st.markdown("### 📤 Upload New Document")
    st.caption(
        "One file at a time, up to 200 MB. Supported: PDF (including scanned), "
        "Word, Excel, PowerPoint, TXT, MD, CSV, RTF and images."
    )

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=SUPPORTED_TYPES,
        accept_multiple_files=False,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if "upload_success" in st.session_state:
        st.success(st.session_state.pop("upload_success"))

    if st.button(
        "🚀 Upload & Index File",
        use_container_width=True,
        disabled=(uploaded_file is None)
    ):
        with st.spinner("Processing file, please wait... (scanned files can take a few minutes)"):
            try:
                upload_response = requests.post(
                    f"{API_URL}/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "application/octet-stream"
                        )
                    },
                    timeout=UPLOAD_TIMEOUT
                )

                if upload_response.status_code == 200:
                    result = upload_response.json()
                    message = (
                        f"Successfully indexed {result['filename']}! "
                        f"{result['total_pages']} {result.get('unit_label', 'pages')}, "
                        f"{result['total_chunks']} chunks."
                    )
                    for note in result.get("notes", []):
                        message += f"\n\n{note}"
                    st.session_state["upload_success"] = message
                    st.session_state.uploader_key += 1
                    st.rerun()
                else:
                    st.error(error_detail(upload_response, "File upload failed."))

            except requests.exceptions.RequestException as e:
                st.error(f"Upload connection failed: {e}")


    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("DocuMind AI • Version 1.0")


# ---------------- MAIN HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>Intelligence from Your Documents ✨</h1>
    <p>Ask questions, discover insights, and get answers directly from your document knowledge base.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 💬 Chat with your documents")
st.caption("Ask a question and DocuMind will search your documents for relevant information.")


# ---------------- CHAT HISTORY ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 View Sources"):
                for source in message["sources"]:
                    st.markdown(source_card(source), unsafe_allow_html=True)


# ---------------- CHAT INPUT ----------------

question = st.chat_input("Ask anything about your documents...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents and generating answer..."):

            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "question": question,
                        "k": 4,
                        "source_filter": (
                            None if selected_source == "All Documents"
                            else selected_source
                        ),
                        "history": [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages[:-1][-6:]
                        ]
                    },
                    timeout=CHAT_TIMEOUT
                )

                if response.status_code == 200:
                    result = response.json()
                    answer = result["answer"]
                    sources = result.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander("📚 View Sources"):
                            for source in sources:
                                st.markdown(source_card(source), unsafe_allow_html=True)

                else:
                    answer = error_detail(response, "Something went wrong.")
                    sources = []
                    st.error(answer)

            except requests.exceptions.RequestException:
                answer = "Could not connect to DocuMind API."
                sources = []
                st.error(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
