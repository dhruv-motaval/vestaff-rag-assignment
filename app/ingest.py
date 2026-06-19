import fitz
import os
from app.config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_PATH
from sentence_transformers import SentenceTransformer
import chromadb
import re
from fastapi import HTTPException, status

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(
    name="document_collection", metadata={"hnsw:space": "cosine"}
)


def extract_text_from_pdf(pdf_path: str):
    """Extracts text page-by-page from a PDF using PyMuPDF. Returns a list of dicts with 'text' and 'page' keys."""

    file_extension = os.path.splitext(pdf_path)[1].lower()

    if file_extension == ".pdf":
        doc = fitz.open(pdf_path)

        pages = []

        for page_number, page in enumerate(doc):
            pages.append({"text": page.get_text(), "page": page_number + 1})

        doc.close()

        return pages

    else:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file extension: {file_extension}",
        )


def clean_text(text):
    """
    Removes URLs, timestamps, page navigation noise, and excess whitespace
    from extracted PDF text before chunking.
    """
    # remove repeated page header
    text = re.sub(r"AWS Customer Agreement", "", text)
    # remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # remove timestamps like 6/16/26, 12:40 PM
    text = re.sub(r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(AM|PM)", "", text)
    # remove page nav like 5/19 or 17/19
    text = re.sub(r"\b\d{1,2}/\d{1,2}\b", "", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Splits text into overlapping chunks of fixed size. Overlap ensures context isn't lost at chunk boundaries."""

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap

    return chunks


def prepare_chunks():
    """Extracts and cleans text from the PDF, then chunks each page. Returns list of dicts with 'context' and 'page'."""

    pages = extract_text_from_pdf(PDF_PATH)

    chunk_records = []

    for page in pages:
        cleaned_text = clean_text(page["text"])

        chunks = chunk_text(cleaned_text)

        for chunk in chunks:
            chunk_records.append({"context": chunk, "page": page["page"]})

    return chunk_records


def store_chunks(chunk_records):
    """Encodes chunks using sentance-transformers and stores documents, embeddings, and metadata in ChromaDB."""

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for idx, chunk in enumerate(chunk_records):
        documents.append(chunk["context"])

        metadatas.append({"page": chunk["page"]})

        ids.append(str(idx))

    embeddings = embedding_model.encode(documents, show_progress_bar=True)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )


def ingest_document():
    """Entry point for ingestion. Skips if collection already has data. Otherwise runs full extract -> chunk -> embed -> store pipeline."""

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Pdf not found : {PDF_PATH}")

    try:
        existing_count = collection.count()

        if existing_count > 0:
            return {
                "status": "already_ingested",
                "chunks": existing_count,
            }
        chunks_records = prepare_chunks()

        store_chunks(chunks_records)

        return {
            "status": "success",
            "chunks_created": len(chunks_records),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
