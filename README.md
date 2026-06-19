# RAG-Based Document Q&A System

A Retrieval-Augmented Generation (RAG) system that answers questions about the AWS Customer Agreement using semantic search and LLM-powered question answering.

Built with **FastAPI**, **ChromaDB**, **Sentence Transformers**, **Groq**, **SQLite**, and **Streamlit**.

---

# Architecture

```text
PDF Document
    │
    ▼
PyMuPDF (Text Extraction)
    │
    ▼
Text Cleaning + Chunking
    │
    ▼
Sentence Transformers
(all-MiniLM-L6-v2)
    │
    ▼
Embeddings
    │
    ▼
ChromaDB (Vector Store)
    │
    ▼
User Query
    │
    ▼
Query Embedding
    │
    ▼
Similarity Search (Top-K)
    │
    ▼
Retrieved Chunks
    │
    ▼
Groq Llama-3.3-70B
    │
    ▼
Answer + Sources
    │
    ▼
FastAPI
    │
    ▼
SQLite Logging
    │
    ▼
Analytics Dashboard
```

---

# Features

* PDF document ingestion
* Semantic search using embeddings
* Retrieval-Augmented Generation (RAG)
* Source citations with page numbers
* FastAPI REST API
* ChromaDB vector storage
* SQLite query logging
* Analytics endpoint
* Streamlit user interface
* Similarity score reporting
* Hallucination prevention through retrieval grounding

---

# Design Decisions

## Chunk Size: 1000 Characters

The AWS Customer Agreement contains long legal clauses and dense paragraphs. A chunk size of 1000 characters preserves enough context for complete contractual clauses while maintaining efficient retrieval.

## Chunk Overlap: 200 Characters

A 200-character overlap reduces information loss at chunk boundaries and improves retrieval quality when relevant content spans multiple chunks.

## Top-K Retrieval: 4

Retrieving the top 4 chunks provides sufficient context for most legal questions while avoiding excessive noise in the prompt.

## Embedding Model: all-MiniLM-L6-v2

A lightweight and efficient embedding model that performs well for semantic similarity search while running locally without external API costs.

## LLM: Groq Llama-3.3-70B Versatile

Chosen for:

* Fast inference speed
* Strong instruction following
* Good performance on retrieval-augmented tasks
* Free developer tier availability

## Vector Store: ChromaDB

ChromaDB provides:

* Persistent local storage
* Fast similarity search
* Minimal operational overhead
* Easy Python integration

Cosine similarity is used for retrieval.

## Hallucination Prevention

The model is instructed to answer strictly from retrieved context.

If the answer is not supported by the retrieved document chunks, the system returns:

```text
Information not found in document.
```

This ensures responses remain grounded in the source document.

---

# Project Structure

```text
vestaff-rag-assignment/
│
├── app/
│   ├── analytics.py
│   ├── config.py
│   ├── db.py
│   ├── ingest.py
│   ├── main.py
│   ├── rag.py
│   └── schemas.py
│
├── chroma_db/
│
├── data/
│   └── AWS Customer Agreement.pdf
│
├── logs.db
│
├── streamlit_app.py
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/vestaff-rag-assignment.git
cd vestaff-rag-assignment
```

## 2. Create Virtual Environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key from:

https://console.groq.com

---

# Running the Application

Run the backend and frontend separately.

## Terminal 1 — FastAPI Backend

```bash
uvicorn app.main:app --reload
```

Backend URL:

```text
http://localhost:8000
```

Swagger Documentation:

```text
http://localhost:8000/docs
```

---

## Terminal 2 — Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

Frontend URL:

```text
http://localhost:8501
```

---

# Usage

### Step 1

Open the Streamlit application.

### Step 2

Click **Ingest Document** to process the AWS Customer Agreement.

The system will:

* Extract text
* Create chunks
* Generate embeddings
* Store vectors in ChromaDB

### Step 3

Ask questions about the agreement.

Example:

```text
What are the customer's responsibilities?
```

### Step 4

Review:

* Generated answer
* Source snippets
* Page references
* Similarity score

### Step 5

Open Analytics to view:

* Most frequently asked questions
* Questions with no answers found
* Average latency
* P95 latency

---

# API Endpoints

| Method | Endpoint     | Description                            |
| ------ | ------------ | -------------------------------------- |
| GET    | `/`          | API status                             |
| GET    | `/health`    | Health check                           |
| POST   | `/ingest`    | Parse, chunk, embed and store document |
| POST   | `/ask`       | Submit question and retrieve answer    |
| GET    | `/analytics` | Usage analytics                        |

---

# Example Request

## POST /ask

```json
{
  "question": "What are the customer's responsibilities?"
}
```

## Example Response

```json
{
  "answer": "The customer is responsible for all activities under their account...",
  "answer_found": true,
  "sources": [
    {
      "page": 2,
      "snippet": "You are responsible for Your Content..."
    }
  ],
  "similarity_score": 0.3254,
  "distance": 0.6746
}
```

---

# Analytics

Every question is logged to SQLite with:

* Question text
* Answer found status
* Response latency
* Timestamp

The analytics endpoint provides:

* Top 10 most frequent questions
* Questions with no answers found
* Average latency
* P95 latency

---

# Future Improvements

* Multi-document support
* Hybrid search (BM25 + Vector Search)
* MultiQueryRetriever
* User feedback collection
* Authentication
* Conversation memory
* Document upload through UI
* Retrieval re-ranking

---

# Tech Stack

* FastAPI
* Streamlit
* ChromaDB
* SQLite
* Groq
* Llama 3.3 70B Versatile
* Sentence Transformers
* PyMuPDF
* NumPy
* Pydantic
