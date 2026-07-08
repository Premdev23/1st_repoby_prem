"""
Argument Builder — RAG-based Legal Argument Generator
======================================================
Uses ChromaDB similarity search to find relevant past cases,
then feeds them into Groq (llama-3.1-8b-instant) to generate
structured legal arguments — same AI used in ingest.py.
"""

import os
import json
from collections import defaultdict
from groq import Groq
from embedder import embed_text
from vector_store import search_chunks

# ── Configure Groq (same key as ingest.py) ────────────────────────────────────
#groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
#GROQ_MODEL = "llama-3.1-8b-instant"
# Replace with:
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.1-8b-instant"
METADATA_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metadata.json")


def load_metadata_cache() -> dict:
    if os.path.exists(METADATA_CACHE):
        with open(METADATA_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def retrieve_similar_cases(facts: str, top_k: int = 5) -> list[dict]:
    """
    Step 1 of RAG: Retrieve the most relevant cases from ChromaDB
    based on the user's facts.
    """
    query_embedding = embed_text(facts)
    raw_results = search_chunks(query_embedding, top_k=top_k * 20)

    metadata_cache = load_metadata_cache()

    case_map = defaultdict(lambda: {
        "scores": [],
        "best_score": 0.0,
        "excerpts": [],
        "meta": {}
    })

    metadatas = raw_results["metadatas"][0]
    distances = raw_results["distances"][0]
    documents = raw_results["documents"][0]

    for meta, distance, doc in zip(metadatas, distances, documents):
        similarity = round(1 - distance, 4)
        if similarity < 0.25:
            continue
        case_id = meta["case_id"]
        case_map[case_id]["scores"].append(similarity)
        case_map[case_id]["meta"] = meta
        if similarity > case_map[case_id]["best_score"]:
            case_map[case_id]["best_score"] = similarity
        if len(case_map[case_id]["excerpts"]) < 3:
            case_map[case_id]["excerpts"].append(doc)

    cases = []
    for case_id, data in case_map.items():
        if not data["scores"]:
            continue

        cached = metadata_cache.get(case_id, {})
        cm = data["meta"]

        def get(key):
            return str(cached.get(key) or cm.get(key) or "")

        cases.append({
            "case_id": case_id,
            "case_name": get("case_name") or case_id,
            "date": get("date"),
            "court": get("court"),
            "citation": get("citation"),
            "description": get("description"),
            "similarity_score": data["best_score"],
            "key_excerpts": data["excerpts"]
        })

    cases.sort(key=lambda x: x["similarity_score"], reverse=True)
    return cases[:top_k]


def build_context_from_cases(cases: list[dict]) -> str:
    """
    Step 2 of RAG: Format retrieved cases into a context string for Groq.
    """
    if not cases:
        return "No similar cases found in the database."

    context_parts = []
    for i, case in enumerate(cases, 1):
        excerpts_text = "\n".join(
            f'  - "{exc[:300]}"' for exc in case["key_excerpts"]
        )
        context_parts.append(f"""
CASE {i}: {case["case_name"]}
  Citation   : {case["citation"]}
  Court      : {case["court"]}
  Date       : {case["date"]}
  Summary    : {case["description"][:400]}
  Similarity : {case["similarity_score"]}
  Key Excerpts:
{excerpts_text}
""")

    return "\n".join(context_parts)


def generate_arguments(facts: str, similar_cases: list[dict]) -> dict:
    """
    Step 3 of RAG: Feed facts + retrieved cases into Groq
    and generate structured legal arguments.
    """
    context = build_context_from_cases(similar_cases)

    prompt = f"""You are an expert Indian legal advocate with deep knowledge of:
- Indian Penal Code (IPC)
- Criminal Procedure Code (CrPC)  
- Indian Evidence Act
- Supreme Court and High Court precedents

A lawyer has presented the following facts:

FACTS OF THE CASE:
{facts}

Most similar past Indian Supreme Court cases from our database:

SIMILAR PAST CASES:
{context}

Using the facts and similar cases above, generate a comprehensive structured legal argument.
Return ONLY a valid JSON object in this exact format with no preamble or markdown:

{{
  "arguments_for": [
    {{
      "point": "Short argument heading",
      "explanation": "Detailed legal reasoning for prosecution/plaintiff",
      "supporting_case": "Case name and citation if applicable, else empty string"
    }}
  ],
  "arguments_against": [
    {{
      "point": "Short argument heading",
      "explanation": "Detailed legal reasoning for defence",
      "supporting_case": "Case name and citation if applicable, else empty string"
    }}
  ],
  "counter_arguments": [
    {{
      "point": "Counter-argument heading",
      "explanation": "How to counter the opposing side's likely argument"
    }}
  ],
  "relevant_laws": [
    {{
      "section": "Section number and act name e.g. Section 302 IPC",
      "description": "What this section covers and why it applies to these facts"
    }}
  ],
  "similar_cases_summary": [
    {{
      "case_name": "Full case name",
      "citation": "Citation e.g. 2025 INSC 221",
      "relevance": "How this case is relevant to the current facts",
      "outcome": "What the court decided in this case"
    }}
  ],
  "overall_assessment": "2-3 sentence overall legal assessment of the case strength for both sides"
}}"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert Indian legal advocate. Always respond with valid JSON only. No markdown, no preamble, no explanation outside the JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,      # Lower = more consistent legal reasoning
        max_tokens=4000,
    )

    raw_text = response.choices[0].message.content.strip()

    # Strip markdown code fences if Groq adds them
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    return json.loads(raw_text)