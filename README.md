# Atlas AI — AI-Powered RAG Chat Assistant

A portfolio-ready full-stack document chat application built with React + Vite, FastAPI, LangChain, ChromaDB and OpenAI.

## What it does

Upload PDF, DOCX, TXT or CSV files, index their content into a persistent ChromaDB vector store, and ask questions through a ChatGPT-style interface.

The RAG pipeline retrieves relevant document chunks before the LLM answers. The system prompt explicitly tells the model to refuse unsupported questions instead of inventing facts.

## Architecture

```text
React + Vite
    │
    │ HTTP / JSON + multipart upload
    ▼
FastAPI
    │
    ├── /upload ──► parse + chunk ──► OpenAI embeddings ──► ChromaDB
    │
    └── /chat ────► embedding search ─► relevant chunks
                                      │
                                      ▼
                                  LangChain
                                      │
                                      ▼
                                  OpenAI LLM
                                      │
                                      ▼
                         grounded answer + sources
                                      │
                                      ▼
                                   React UI
```

## Project structure

```text
ai-rag-assistant/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── .env.example
│   ├── app/
│   │   └── __init__.py
│   └── data/
│       └── .gitkeep
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── services/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── package.json
│   ├── index.html
│   └── vite.config.js
├── .gitignore
└── README.md
```

## 1. Requirements on Windows

Install these first:

- Python 3.11 or newer
- Node.js 20 LTS or newer
- Git
- VS Code

Verify:

```powershell
python --version
node --version
npm --version
git --version
```

## 2. Open the project in VS Code

In PowerShell:

```powershell
cd path\to\ai-rag-assistant
code .
```

Or open VS Code manually and select the `ai-rag-assistant` folder.

## 3. Backend setup

Open a VS Code terminal:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Create `.env`

Copy `.env.example` to `.env`.

PowerShell:

```powershell
Copy-Item .env.example .env
```

Open `.env` and set:

```env
OPENAI_API_KEY=your_real_openai_key
FRONTEND_ORIGIN=http://localhost:5173
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-5.6-luna
```

Never commit `.env`.

## 4. Start the backend

From `backend`:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend:

```text
http://localhost:8000
```

API documentation:

```text
http://localhost:8000/docs
```

Health check:

```text
http://localhost:8000/health
```

## 5. Frontend setup

Open a second VS Code terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

The React app calls FastAPI at `http://localhost:8000`.

### Optional frontend `.env`

If your backend runs somewhere else, create `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000
```

Do not put the OpenAI API key in the frontend.

## 6. How the RAG pipeline works

### Upload

1. React sends the file as multipart form data to `POST /upload`.
2. FastAPI validates the extension and 20 MB limit.
3. The backend extracts text:
   - PDF → pypdf, one source document per page
   - DOCX → python-docx
   - TXT → UTF-8 text
   - CSV → rows converted into searchable text
4. LangChain's `RecursiveCharacterTextSplitter` creates overlapping chunks.
5. OpenAI embeddings convert chunks into vectors.
6. ChromaDB stores the vectors persistently in `backend/data/chroma`.

### Question

1. React sends the question to `POST /chat`.
2. The question is embedded by the same embedding model.
3. ChromaDB retrieves the five most relevant chunks.
4. Only sufficiently relevant chunks are passed to the LLM.
5. LangChain formats a strict grounding prompt.
6. The LLM answers from the supplied context.
7. The API returns the answer and source document/page metadata.
8. React displays the answer and source chips.

## 7. Grounding / hallucination protection

This application uses several layers:

- Similarity retrieval before generation
- Relevance threshold
- Small retrieval set
- Temperature 0
- Explicit system instructions to use only supplied context
- Explicit refusal when context is insufficient
- Source metadata attached to retrieved chunks
- Prompt-injection defense: document text is treated as untrusted data

No RAG system can mathematically guarantee zero hallucinations, but these controls substantially reduce unsupported answers.

## 8. API endpoints

### GET `/health`

Returns backend and OpenAI configuration status.

### POST `/upload`

Multipart upload:

```text
file=<PDF/DOCX/TXT/CSV>
```

### GET `/documents`

Returns indexed documents.

### POST `/chat`

JSON:

```json
{
  "message": "What are the main requirements?",
  "conversation_id": "optional-id"
}
```

### GET `/conversations`

Returns saved conversation history.

### POST `/conversations`

JSON:

```json
{
  "title": "Project requirements"
}
```

### DELETE `/conversations/{id}`

Deletes one conversation.

### GET `/analytics`

Returns document count, question count, conversation count and document type distribution.

## 9. Testing checklist

### Backend

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn main:app --reload --port 8000
```

Visit:

```text
http://localhost:8000/health
```

You should see `"status": "ok"`.

### Frontend

```powershell
cd frontend
npm run dev
```

Open the Vite URL.

### Functional test

1. Upload a small text/PDF document.
2. Wait for "indexed successfully".
3. Ask a question whose answer is definitely in the document.
4. Confirm an answer appears.
5. Confirm a source chip shows document name and page.
6. Ask something unrelated.
7. The assistant should explain that it cannot find enough information.
8. Refresh the browser.
9. Confirm the conversation remains visible.
10. Open Dashboard and verify document/question counts.
11. Switch light/dark mode.
12. Test drag-and-drop upload.
13. Test Copy and Regenerate.

## 10. Troubleshooting

### `ModuleNotFoundError`

Activate the virtual environment and reinstall:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### `python is not recognized`

Install Python and enable "Add Python to PATH", then reopen VS Code.

### `npm is not recognized`

Install Node.js LTS and reopen VS Code.

### `OPENAI_API_KEY is not configured`

Make sure:

```text
backend/.env
```

exists and contains:

```env
OPENAI_API_KEY=...
```

Restart Uvicorn after changing `.env`.

### CORS error

Make sure:

```env
FRONTEND_ORIGIN=http://localhost:5173
```

matches the actual Vite URL. Then restart the backend.

FastAPI requires explicit allowed origins when frontend and backend run on different origins.

### Upload fails

Check:

- File is PDF, DOCX, TXT or CSV
- File is <= 20 MB
- The document contains selectable/readable text
- Backend terminal for the exact exception

Scanned image-only PDFs are not OCR'd by this version.

### Chroma error / corrupted index

Stop the backend and remove:

```text
backend/data/chroma
```

Then restart and re-upload your documents.

### Port already in use

Backend:

```powershell
uvicorn main:app --reload --port 8001
```

Then set frontend:

```env
VITE_API_URL=http://localhost:8001
```

For a different Vite port:

```powershell
npm run dev -- --port 5174
```

Then update:

```env
FRONTEND_ORIGIN=http://localhost:5174
```

### PowerShell execution policy

Run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then reopen the terminal.

## 11. Production notes

This is a strong portfolio/demo implementation. Before public production deployment, add:

- User authentication
- Per-user document collections
- Database-backed conversations instead of JSON
- Background document jobs / queue
- Object storage for uploads
- Rate limiting
- File malware scanning
- OCR for scanned PDFs
- Streaming LLM responses
- Observability and structured logging
- HTTPS
- Secret manager
- Stronger tenant isolation
- Automated tests and CI/CD

## 12. GitHub

From the project root:

```powershell
git init
git add .
git commit -m "Build AI RAG chat assistant"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Because `.env`, Chroma data, uploaded files and local runtime state are ignored, your OpenAI key and user documents will not be committed.

## License

Use this project as a portfolio foundation and customize the branding, retrieval strategy, authentication and deployment for your needs.
