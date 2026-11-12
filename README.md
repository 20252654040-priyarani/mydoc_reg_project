# DocuMind - Multi-Document AI Knowledge Assistant

DocuMind is a Retrieval-Augmented Generation (RAG) application. Upload documents and ask questions; answers come only from your files, with the file and page / sheet / slide shown as the source.

## Features
- One file per upload, up to 200 MB
- Supported files: PDF, scanned PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx), TXT, MD, CSV, RTF and images (PNG, JPG, WEBP, BMP, TIFF)
- OCR for scanned PDFs and images using Gemini vision
- Long-document handling: chunking with overlap, batched embeddings with retries
- Google Gemini embeddings + FAISS vector database
- MMR-based retrieval
- Source citations: page, sheet, slide, section or row range
- Search all documents or a single document
- Chat memory: follow-up questions are rewritten into standalone search queries
- Clear errors for password-protected, corrupt, empty, oversized and unsupported files
- Secure uploads: safe file names, extension + content checks, size limits, duplicate protection
- Health check endpoint and server logging
- FastAPI backend (Render) and Streamlit frontend (Streamlit Community Cloud)

## Project Files
- app.py: FastAPI backend
- frontend.py: Streamlit frontend
- requirements.txt: Dependencies
- README.md: Documentation

## Backend settings (environment variables)
- GOOGLE_API_KEY (required): Gemini API key. Never upload it to GitHub.
- GEMINI_MODEL (optional, default gemini-3.5-flash-lite)
- EMBEDDING_MODEL (optional, default models/gemini-embedding-001)
- MAX_UPLOAD_MB (default 200), MAX_OCR_PAGES (default 60), MAX_CHUNKS (default 6000)

## Run locally
- Backend: uvicorn app:app --reload --port 8000
- Frontend: BACKEND_URL=http://127.0.0.1:8000 streamlit run frontend.py

## Deployment
- Render: Build command `pip install -r requirements.txt`, start command `uvicorn app:app --host 0.0.0.0 --port $PORT`, health check path `/health`, environment variable GOOGLE_API_KEY.
- Streamlit Community Cloud: main file `frontend.py`. Optional secret: BACKEND_URL (the Render link).

## API Endpoints
- GET / : API information
- GET /health : Health check (documents, index status, limits, formats)
- GET /documents : List documents
- POST /upload : Upload and index one file
- POST /retrieve : Retrieve relevant chunks
- POST /chat : Ask document-based questions

## Limits on the free Render plan
- The free server has 512 MB RAM and sleeps after 15 minutes of inactivity (first request takes about a minute).
- Uploaded files are stored temporarily and are cleared when the server restarts.
- Scanned files: the first 60 scanned pages of a file are read with OCR; the rest are skipped.

## Security
- Keep API keys private.
- Uploaded content is treated as untrusted input.
- Do not commit private documents or FAISS database files.
