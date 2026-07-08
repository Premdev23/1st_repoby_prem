"""
Legal Case Similarity Finder — FastAPI Backend
===============================================
Endpoints:
  POST /find-similar              → Search similar cases by query text
  POST /build-arguments           → Build legal arguments from case facts (RAG)
  GET  /cases/download/{case_id}  → Download original PDF
  GET  /cases/view/{case_id}      → View PDF inline in browser
  GET  /stats                     → DB statistics
  GET  /                          → Health check
"""

import os
import json
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from models import SimilarCaseQuery, CaseResult, ArgumentRequest, ArgumentResponse
from embedder import embed_text
from vector_store import search_chunks, get_chunk_count
from argument_builder import retrieve_similar_cases, generate_arguments
from typing import List

app = FastAPI(
    title="Legal Case Similarity Finder",
    version="2.0.0",
    description="Find similar Indian legal cases using AI semantic search (Gemini Flash + ChromaDB)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ──────────────────────────────────────────────────────────────────────
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#PDF_DIR = os.path.join(BASE_DIR, "files", "archive", "supreme_court_judgments", "2025")
PDF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "files", "archive", "supreme_court_judgments", "2025")
METADATA_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metadata.json")


def load_metadata_cache() -> dict:
    if os.path.exists(METADATA_CACHE):
        with open(METADATA_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def find_pdf(case_id: str):
    """Locate PDF with case-insensitive fallback. Returns full path or None."""
    if not os.path.isdir(PDF_DIR):
        return None
    direct_path = os.path.join(PDF_DIR, f"{case_id}.pdf")
    if os.path.exists(direct_path):
        return direct_path
    target = f"{case_id.lower()}.pdf"
    for fname in os.listdir(PDF_DIR):
        if fname.lower() == target:
            return os.path.join(PDF_DIR, fname)
    return None


def aggregate_by_case(results, top_k, min_score, base_url) -> List[CaseResult]:
    """Aggregate chunk-level similarity scores up to case level."""
    metadata_cache = load_metadata_cache()
    case_map = defaultdict(lambda: {"scores": [], "best_score": 0.0, "best_excerpt": "", "meta": {}})

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    documents = results["documents"][0]
    ids = results["ids"][0]

    for meta, distance, doc, chunk_id in zip(metadatas, distances, documents, ids):
        similarity = round(1 - distance, 4)
        if similarity < min_score:
            continue
        case_id = meta["case_id"]
        case_map[case_id]["scores"].append(similarity)
        case_map[case_id]["meta"] = meta
        if similarity > case_map[case_id]["best_score"]:
            case_map[case_id]["best_score"] = similarity
            case_map[case_id]["best_excerpt"] = doc

    aggregated = []
    for case_id, data in case_map.items():
        if not data["scores"]:
            continue
        matched_chunks = len(data["scores"])
        boost = min(0.01 * (matched_chunks - 1), 0.05)
        final_score = round(min(data["best_score"] + boost, 1.0), 4)

        cached = metadata_cache.get(case_id, {})
        cm = data["meta"]

        def get(key):
            return str(cached.get(key) or cm.get(key) or "")

        aggregated.append(CaseResult(
            case_id=case_id,
            case_name=get("case_name") or case_id,
            date=get("date"),
            court=get("court"),
            judges=get("judges"),
            case_number=get("case_number"),
            citation=get("citation"),
            appellant=get("appellant"),
            respondent=get("respondent"),
            description=get("description"),
            similarity_score=final_score,
            best_matching_excerpt=data["best_excerpt"][:600],
            matched_chunks=matched_chunks,
            download_url=f"{base_url}/cases/download/{case_id}",
            view_url=f"{base_url}/cases/view/{case_id}"
        ))

    aggregated.sort(key=lambda x: x.similarity_score, reverse=True)
    return aggregated[:top_k]


# ── POST /find-similar ─────────────────────────────────────────────────────────
@app.post("/find-similar", response_model=List[CaseResult], summary="Find similar legal cases")
async def find_similar_cases(query: SimilarCaseQuery, request: Request):
    """Search for similar legal cases using semantic AI embeddings."""
    total = get_chunk_count()
    if total == 0:
        raise HTTPException(status_code=503, detail="Database is empty. Run `python ingest.py` first.")

    fetch_k = min(query.top_k * 25, total)
    query_embedding = embed_text(query.query_text)
    raw_results = search_chunks(query_embedding, top_k=fetch_k)

    base_url = str(request.base_url).rstrip("/")
    similar_cases = aggregate_by_case(raw_results, top_k=query.top_k, min_score=query.min_score, base_url=base_url)

    if not similar_cases:
        raise HTTPException(status_code=404, detail=f"No cases found above similarity {query.min_score}.")

    return similar_cases


# ── POST /build-arguments ──────────────────────────────────────────────────────
@app.post("/build-arguments", response_model=ArgumentResponse, summary="Build legal arguments from case facts")
async def build_arguments(request: ArgumentRequest):
    """
    Generate structured legal arguments using RAG (Retrieval Augmented Generation).

    How it works:
    1. Your facts are embedded and searched against the case database
    2. The most similar past cases are retrieved as context
    3. Gemini uses your facts + those cases to generate structured arguments

    Input:
    - facts: The facts of your case (be as detailed as possible)
    - top_k_cases: How many similar cases to use as context (default: 5)

    Output:
    - Arguments FOR (prosecution/plaintiff)
    - Arguments AGAINST (defence)
    - Counter-arguments
    - Relevant IPC sections and laws
    - Summary of similar past cases used
    - Overall legal assessment
    """
    total = get_chunk_count()
    if total == 0:
        raise HTTPException(status_code=503, detail="Database is empty. Run `python ingest.py` first.")

    # Step 1: Retrieve similar cases from ChromaDB
    similar_cases = retrieve_similar_cases(request.facts, top_k=request.top_k_cases)

    if not similar_cases:
        raise HTTPException(status_code=404, detail="No similar cases found. Try providing more detailed facts.")

    # Step 2 + 3: Generate arguments via Gemini
    try:
        result = generate_arguments(request.facts, similar_cases)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Argument generation failed: {str(e)}")

    return ArgumentResponse(**result, cases_used=len(similar_cases))


# ── GET /cases/download/{case_id} ─────────────────────────────────────────────
@app.get("/cases/download/{case_id}", summary="Download original PDF")
def download_case_pdf(case_id: str):
    """Download the original PDF file for a given case."""
    pdf_path = find_pdf(case_id)
    if not pdf_path:
        if not os.path.isdir(PDF_DIR):
            raise HTTPException(status_code=500, detail=f"PDF directory not found: {PDF_DIR}")
        raise HTTPException(status_code=404, detail=f"PDF not found for case: {case_id}")
    return FileResponse(path=pdf_path, media_type="application/pdf", filename=f"{case_id}.pdf",
                        headers={"Content-Disposition": f'attachment; filename="{case_id}.pdf"'})


# ── GET /cases/view/{case_id} ──────────────────────────────────────────────────
@app.get("/cases/view/{case_id}", summary="View PDF inline in browser")
def view_case_pdf(case_id: str):
    """View the PDF inline in the browser."""
    pdf_path = find_pdf(case_id)
    if not pdf_path:
        if not os.path.isdir(PDF_DIR):
            raise HTTPException(status_code=500, detail=f"PDF directory not found: {PDF_DIR}")
        raise HTTPException(status_code=404, detail=f"PDF not found for case: {case_id}")
    return FileResponse(path=pdf_path, media_type="application/pdf", filename=f"{case_id}.pdf",
                        headers={"Content-Disposition": f'inline; filename="{case_id}.pdf"'})


# ── GET /stats ─────────────────────────────────────────────────────────────────
@app.get("/stats", summary="Database statistics")
def stats():
    metadata_cache = load_metadata_cache()
    return {
        "total_cases_indexed": len(metadata_cache),
        "total_chunks_indexed": get_chunk_count(),
        "pdf_directory": PDF_DIR,
        "pdf_directory_exists": os.path.isdir(PDF_DIR),
        "ai_model": "Gemini 1.5 Flash (arguments + descriptions) + all-MiniLM-L6-v2 (embeddings)",
        "status": "ready" if get_chunk_count() > 0 else "empty — run python ingest.py"
    }


# ── GET / ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "⚖️ Legal Case Similarity Finder API v2.0",
        "docs": "/docs",
        "endpoints": {
            "find_similar": "POST /find-similar",
            "build_arguments": "POST /build-arguments",
            "download_pdf": "GET /cases/download/{case_id}",
            "view_pdf": "GET /cases/view/{case_id}",
            "stats": "GET /stats"
        },
        "status": "running"
    }