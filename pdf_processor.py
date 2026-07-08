import fitz  # PyMuPDF
import re
import os
import time
from groq import Groq

# ── Groq Setup ─────────────────────────────────────────────────────────────────
# Get free API key at: https://console.groq.com
# Set env variable:  set GROQ_API_KEY=your_key_here        (Windows CMD)
#                    export GROQ_API_KEY="your_key_here"   (Linux/Mac)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ══════════════════════════════════════════════════════════════════════════════
# PDF TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a multi-page PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n"
    doc.close()
    return full_text.strip()


def clean_text(text: str) -> str:
    """Remove noise, excessive whitespace, page numbers, footers."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'Indian Kanoon.*?\n', '', text)
    text = re.sub(r'http\S+', '', text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# AI DESCRIPTION — GROQ (FREE TIER, VERY FAST)
# Free tier: 14,400 requests/day, 30 requests/minute on llama-3.1-8b-instant
# ══════════════════════════════════════════════════════════════════════════════

def generate_ai_description(text: str, retries: int = 3) -> str:
    """
    Use Groq (Llama 3.1 8B) to generate a clean 2-3 sentence case description.
    - Extremely fast: 400 PDFs done in ~5-8 minutes
    - Auto-retries on rate limit with backoff
    - Falls back to regex extraction if all retries fail
    """
    prompt = f"""You are a legal assistant specializing in Indian court cases.

Read this court case excerpt and write a clear 2-3 sentence summary covering:
1. Who the parties are (appellant vs respondent)
2. What the core legal dispute or issue is about
3. What the court's outcome or verdict was (if mentioned)

Rules:
- Be factual and concise (under 150 words)
- No opinions or analysis
- Use plain English, not excessive legal jargon
- If verdict is not mentioned, just summarize the dispute
- Return ONLY the summary, no extra text

Court case excerpt:
{text[:2500]}

Summary:"""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",   # Free, very fast model
                messages=[
                    {
                        "role": "system",
                        "content": "You are a legal assistant. Write concise, factual summaries of Indian court cases."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=200,
                temperature=0.1   # Low = consistent, factual output
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            error_str = str(e).lower()

            if "rate_limit" in error_str or "429" in error_str or "rate limit" in error_str:
                wait = 30 if attempt == 0 else 60
                print(f"   Rate limit hit. Waiting {wait}s (retry {attempt+1}/{retries})...")
                time.sleep(wait)

            elif "api_key" in error_str or "authentication" in error_str or "invalid" in error_str or "auth" in error_str:
                print("Invalid Groq API key! Set GROQ_API_KEY environment variable.")
                print("Get free key at: https://console.groq.com")
                break

            elif "model" in error_str:
                print(" Model error, trying fallback model...")
                try:
                    # Fallback to another free Groq model
                    response = client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200,
                        temperature=0.1
                    )
                    return response.choices[0].message.content.strip()
                except:
                    pass

            else:
                print(f"    Groq error attempt {attempt+1}/{retries}: {e}")
                time.sleep(5)

    # All retries failed — fall back to regex
    print(" Groq failed, using regex fallback...")
    return extract_regex_description(text)


def extract_regex_description(text: str) -> str:
    """
    Fallback: extract description using regex when Groq is unavailable.
    Finds the first substantive paragraph after JUDGMENT/ORDER keyword.
    """
    judgment_match = re.search(
        r'(?:JUDGMENT|ORDER|JUDGEMENT)\s*\n(.+?)(?:\n\n|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    if judgment_match:
        raw = re.sub(r'\s+', ' ', judgment_match.group(1).strip())
        if len(raw) >= 100:
            return raw[:500]

    # Fallback: skip header lines, grab body text
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    body = ' '.join(lines[10:25])
    return re.sub(r'\s+', ' ', body)[:500]


# ══════════════════════════════════════════════════════════════════════════════
# METADATA EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_metadata_from_text(text: str, filename: str) -> dict:
    """
    Extract structured metadata from Indian Supreme/High Court legal PDFs.
    Handles standard Indian legal PDF format:
    e.g. "Akshay Gupta vs ICICI Bank Limited on 25 March, 2025"
    """
    metadata = {
        "case_name": "",
        "date": "",
        "court": "",
        "judges": "",
        "case_number": "",
        "citation": "",
        "description": "",
        "appellant": "",
        "respondent": ""
    }

    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # ── Case Name ──────────────────────────────────────────────────────────────
    case_name_match = re.search(
        r'^(.+?)\s+on\s+\d{1,2}\s+\w+,?\s+\d{4}',
        text[:500], re.MULTILINE | re.IGNORECASE
    )
    if case_name_match:
        metadata["case_name"] = case_name_match.group(1).strip()
    else:
        metadata["case_name"] = lines[0] if lines else os.path.splitext(filename)[0]

    # ── Date ───────────────────────────────────────────────────────────────────
    date_match = re.search(
        r'on\s+(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December),?\s+\d{4})',
        text[:500], re.IGNORECASE
    )
    if date_match:
        metadata["date"] = date_match.group(1).strip()
    else:
        alt = re.search(r'\b(\d{1,2}[./]\d{1,2}[./]\d{4})\b', text[:1000])
        if alt:
            metadata["date"] = alt.group(1)

    # ── Court Name ─────────────────────────────────────────────────────────────
    court_patterns = [
        r'(SUPREME COURT OF INDIA)',
        r'(HIGH COURT OF [A-Z\s]+)',
        r'(NATIONAL CONSUMER DISPUTES? REDRESSAL COMMISSION)',
        r'(DISTRICT CONSUMER DISPUTES REDRESSAL COMMISSION[,\s\w]*)',
        r'(NATIONAL COMPANY LAW TRIBUNAL[,\s\w]*)',
        r'(DEBT RECOVERY TRIBUNAL[,\s\w]*)',
        r'(CENTRAL ADMINISTRATIVE TRIBUNAL[,\s\w]*)',
        r'(SESSIONS COURT[,\s\w]*)',
        r'(CIVIL COURT[,\s\w]*)',
        r'(FAMILY COURT[,\s\w]*)',
    ]
    for pattern in court_patterns:
        match = re.search(pattern, text[:2000], re.IGNORECASE)
        if match:
            metadata["court"] = match.group(1).title().strip()
            break

    if not metadata["court"]:
        generic = re.search(r'IN\s+THE\s+(.+?COURT.+?)\n', text[:2000], re.IGNORECASE)
        if generic:
            metadata["court"] = generic.group(1).strip().title()

    # ── Judges / Bench ─────────────────────────────────────────────────────────
    bench_match = re.search(r'(?:Bench|BENCH)\s*:\s*(.+?)(?:\n|$)', text[:1000], re.IGNORECASE)
    if bench_match:
        metadata["judges"] = bench_match.group(1).strip()
    else:
        author_match = re.search(r'(?:Author|AUTHOR)\s*:\s*(.+?)(?:\n|$)', text[:1000], re.IGNORECASE)
        if author_match:
            metadata["judges"] = author_match.group(1).strip()

    # ── Case Number ────────────────────────────────────────────────────────────
    case_num_match = re.search(
        r'((?:CIVIL|CRIMINAL|WRIT|SPECIAL LEAVE)?\s*'
        r'(?:APPEAL|PETITION|APPLICATION|SUIT|REVISION)'
        r'(?:\s+NO\.?\s*[\w/()]+(?:\s+OF\s+\d{4})?))',
        text[:2000], re.IGNORECASE
    )
    if case_num_match:
        metadata["case_number"] = case_num_match.group(1).strip()

    # ── Citation (e.g. 2025 INSC 391) ─────────────────────────────────────────
    citation_match = re.search(
        r'(\d{4}\s+(?:INSC|SCC|AIR|SCR|HC|DLT|BOM|MAD|CAL|ALL)\s+\d+)',
        text[:1000], re.IGNORECASE
    )
    if citation_match:
        metadata["citation"] = citation_match.group(1).strip()

    # ── Appellant / Respondent ─────────────────────────────────────────────────
    appellant_match = re.search(
        r'([A-Z][A-Z\s&.,()\/]+?)\s*[…\.]+\s*APPELLANTS?',
        text[:3000], re.IGNORECASE
    )
    respondent_match = re.search(
        r'([A-Z][A-Z\s&.,()\/]+?)\s*[…\.]+\s*RESPONDENTS?',
        text[:3000], re.IGNORECASE
    )
    if appellant_match:
        metadata["appellant"] = appellant_match.group(1).strip().title()
    if respondent_match:
        metadata["respondent"] = respondent_match.group(1).strip().title()

    return metadata


# ══════════════════════════════════════════════════════════════════════════════
# TEXT CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """
    Split cleaned text into overlapping word-based chunks.
    - chunk_size = 500 words (~750 tokens, safe for embedding model)
    - overlap   = 100 words shared between consecutive chunks
                  (preserves context at chunk boundaries)
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
        if start >= len(words):
            break
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_pdf(pdf_path: str, case_id: str, use_ai_description: bool = True) -> dict:
    """
    Full pipeline for a single PDF:
      1. Extract raw text (PyMuPDF)
      2. Clean text (remove noise/footers)
      3. Extract structured metadata (regex)
      4. Generate AI description (Groq Llama 3.1 / regex fallback)
      5. Chunk text for vector embedding

    Args:
        pdf_path          : Full path to the PDF file
        case_id           : Unique ID (filename without .pdf extension)
        use_ai_description: False = skip Groq, use regex only (faster)

    Returns:
        dict: { case_id, metadata, chunks, full_text }
    """
    raw_text = extract_text_from_pdf(pdf_path)
    clean = clean_text(raw_text)

    metadata = extract_metadata_from_text(clean, os.path.basename(pdf_path))

    if use_ai_description:
        metadata["description"] = generate_ai_description(clean)
    else:
        metadata["description"] = extract_regex_description(clean)

    chunks = chunk_text(clean)

    return {
        "case_id": case_id,
        "metadata": metadata,
        "chunks": chunks,
        "full_text": clean
    }