"""
ingest.py — Run ONCE to index all 400 PDFs into ChromaDB
=========================================================
- Extracts text from each PDF
- Generates AI description using Gemini Flash (free tier)
- Embeds chunks and stores in ChromaDB
- Auto-resumes if interrupted (skips already indexed cases)
- Handles Gemini rate limit: 15 requests/minute = 4 sec delay between PDFs

Usage:
    python ingest.py                    # Index all PDFs with AI descriptions
    python ingest.py --no-ai            # Skip Gemini, use regex descriptions
    python ingest.py --reset            # Clear DB and re-index everything

Set your API key first:
    export GEMINI_API_KEY="your_key_here"
"""

import os
import sys
import json
import time
from tqdm import tqdm
from pdf_processor import process_pdf
from embedder import embed_batch
from vector_store import add_chunks_batch, get_chunk_count, reset_collection

PDF_DIR = "files/archive/supreme_court_judgments/2025"
METADATA_CACHE = "cloude/metadata.json"

# Groq free tier = 30 requests/minute on llama-3.1-8b-instant
# 60 sec / 30 = 2 seconds minimum between requests
GROQ_RATE_LIMIT_DELAY = 2.5  # seconds between PDFs (safe margin)


def load_cache() -> dict:
    if os.path.exists(METADATA_CACHE):
        with open(METADATA_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(METADATA_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def ingest_all_pdfs(use_ai: bool = True):
    # ── Validate setup ─────────────────────────────────────────────────────────
    if not os.path.exists(PDF_DIR):
        print(f" PDF directory '{PDF_DIR}' not found. Create it and add your PDFs.")
        sys.exit(1)

    pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
    if not pdf_files:
        print(f" No PDF files found in '{PDF_DIR}'.")
        sys.exit(1)

    if use_ai and not os.environ.get("GROQ_API_KEY"):
        print("  WARNING: GROQ_API_KEY not set!")
        print("   Set it with: set GROQ_API_KEY=your_key_here (Windows)")
        print("                export GROQ_API_KEY='your_key_here' (Linux/Mac)")
        print("   Get free key: https://console.groq.com")
        print("   Falling back to regex descriptions...\n")
        use_ai = False
    print(f"{'='*60}")
    print(f"  Legal Case Finder — PDF Ingestion")
    print(f"{'='*60}")
    print(f" PDFs found      : {len(pdf_files)}")
    print(f" AI descriptions : {'Groq Llama 3.1 ' if use_ai else 'Regex fallback'}")
    print(f"  Est. time       : {'~{} mins'.format(len(pdf_files) * GROQ_RATE_LIMIT_DELAY // 60 + 10) if use_ai else '~15-20 mins'}")
    print(f"{'='*60}\n")

    # ── Load existing cache (for resume support) ───────────────────────────────
    metadata_cache = load_cache()
    already_indexed = set(metadata_cache.keys())
    remaining = [f for f in pdf_files if os.path.splitext(f)[0] not in already_indexed]

    if already_indexed:
        print(f" Already indexed : {len(already_indexed)} cases (will skip)")
        print(f" Remaining       : {len(remaining)} PDFs to process\n")

    if not remaining:
        print(" All PDFs already indexed! Nothing to do.")
        print(f"   Total chunks in DB: {get_chunk_count()}")
        return

    # ── Process each PDF ───────────────────────────────────────────────────────
    failed = []
    success_count = 0

    for i, pdf_file in enumerate(tqdm(remaining, desc="Indexing PDFs")):
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        case_id = os.path.splitext(pdf_file)[0]

        try:
            # Step 1: Extract text + metadata + AI description
            result = process_pdf(pdf_path, case_id, use_ai_description=use_ai)
            meta = result["metadata"]
            chunks = result["chunks"]

            if not chunks:
                print(f"\n    No text extracted: {pdf_file}")
                failed.append(pdf_file)
                continue

            # Step 2: Embed all chunks (batch for speed)
            texts = chunks
            embeddings = embed_batch(texts)

            # Step 3: Build ChromaDB metadata (all fields must be strings/ints)
            chunk_ids = [f"{case_id}_chunk_{idx}" for idx in range(len(chunks))]
            chroma_metadatas = [
                {
                    "case_id": case_id,
                    "filename": pdf_file,
                    "case_name": str(meta.get("case_name", ""))[:200],
                    "date": str(meta.get("date", "")),
                    "court": str(meta.get("court", ""))[:200],
                    "judges": str(meta.get("judges", ""))[:200],
                    "case_number": str(meta.get("case_number", ""))[:200],
                    "citation": str(meta.get("citation", "")),
                    "appellant": str(meta.get("appellant", ""))[:200],
                    "respondent": str(meta.get("respondent", ""))[:200],
                    "description": str(meta.get("description", ""))[:400],
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                }
                for idx in range(len(chunks))
            ]

            # Step 4: Save to ChromaDB
            add_chunks_batch(chunk_ids, embeddings, chroma_metadatas, texts)

            # Step 5: Save to metadata cache
            metadata_cache[case_id] = {"filename": pdf_file, **meta}
            save_cache(metadata_cache)  # Save after each PDF (safe resume)

            success_count += 1
            tqdm.write(f"   [{i+1}/{len(remaining)}] {pdf_file[:55]} | {len(chunks)} chunks | {meta.get('case_name', '')[:40]}")

            # Groq rate limiting: 30 requests/min → wait between PDFs
            if use_ai and i < len(remaining) - 1:
                time.sleep(GROQ_RATE_LIMIT_DELAY)

        except KeyboardInterrupt:
            print(f"\n\n Interrupted! Progress saved. Re-run to resume from where you left off.")
            break

        except Exception as e:
            tqdm.write(f"   FAILED: {pdf_file} → {e}")
            failed.append(pdf_file)
            continue

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" Ingestion Complete!")
    print(f"{'='*60}")
    print(f"   Successfully indexed : {success_count} PDFs")
    print(f"   Failed               : {len(failed)} PDFs")
    print(f"   Total cases in DB    : {len(metadata_cache)}")
    print(f"   Total chunks in DB   : {get_chunk_count()}")

    if failed:
        print(f"\n  Failed files:")
        for f in failed:
            print(f"    - {f}")
        print(f"\n  Re-run ingest.py to retry failed files.")

    print(f"\n   Ready! Start API with: uvicorn main:app --reload")


if __name__ == "__main__":
    use_ai = "--no-ai" not in sys.argv

    if "--reset" in sys.argv:
        confirm = input("  This will DELETE all indexed data. Type 'yes' to confirm: ")
        if confirm.lower() == "yes":
            reset_collection()
            if os.path.exists(METADATA_CACHE):
                os.remove(METADATA_CACHE)
            print(" Database cleared.\n")
        else:
            print("Cancelled.")
            sys.exit(0)

    ingest_all_pdfs(use_ai=use_ai)