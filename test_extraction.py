"""
test_extraction.py — Test metadata + AI description on a single PDF before ingesting all 400.

Usage:
    python test_extraction.py path/to/your_case.pdf
    python test_extraction.py path/to/your_case.pdf --no-ai   # Test regex only
"""
import sys
import os
from pdf_processor import process_pdf


def test_pdf(pdf_path: str, use_ai: bool = True):
    case_id = os.path.splitext(os.path.basename(pdf_path))[0]

    print(f"\n{'='*65}")
    print(f"🧪 Testing: {pdf_path}")
    print(f"{'='*65}")

    result = process_pdf(pdf_path, case_id, use_ai_description=use_ai)
    meta = result["metadata"]
    chunks = result["chunks"]

    print(f"\n📋 EXTRACTED METADATA")
    print(f"  {'Case Name':<14}: {meta['case_name']}")
    print(f"  {'Date':<14}: {meta['date']}")
    print(f"  {'Court':<14}: {meta['court']}")
    print(f"  {'Judges':<14}: {meta['judges']}")
    print(f"  {'Case Number':<14}: {meta['case_number']}")
    print(f"  {'Citation':<14}: {meta['citation']}")
    print(f"  {'Appellant':<14}: {meta['appellant']}")
    print(f"  {'Respondent':<14}: {meta['respondent']}")

    print(f"\n🤖 AI DESCRIPTION ({'Gemini Flash' if use_ai else 'Regex fallback'})")
    print(f"  {meta['description']}")

    print(f"\n📦 CHUNKING")
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Chunk 0 preview:")
    print(f"  {chunks[0][:200]}...")

    print(f"\n{'='*65}")
    print(f"✅ Test complete! If metadata looks good, run: python ingest.py")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_extraction.py path/to/file.pdf [--no-ai]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    use_ai = "--no-ai" not in sys.argv

    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    test_pdf(pdf_path, use_ai=use_ai)
