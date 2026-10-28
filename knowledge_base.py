"""Local RAG pipeline: document ingestion, extraction, overlapping chunking, cached lexical index, retrieval.

Supported sources: UTF-8 .md/.txt and PDF files under knowledge_base/. No paid embedding API is required.
"""
from __future__ import annotations
import math, os, re, threading
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = Path(os.getenv('RAG_DOCS_DIR', str(BASE_DIR / 'knowledge_base')))
TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u0900-\u097F]+", re.UNICODE)
STOP = set("a an the and or to of in for is are was were be with on at by from it this that can do does how what when where who why i we you your our store product products order orders customer customers please tell about".split())
_CHUNK_SIZE = max(300, int(os.getenv('RAG_CHUNK_CHARS', '900')))
_CHUNK_OVERLAP = max(0, min(_CHUNK_SIZE // 3, int(os.getenv('RAG_CHUNK_OVERLAP', '120'))))
_cache_lock = threading.Lock()
_cache_signature = None
_cache_chunks = []
_cache_df = Counter()
_cache_idf = {}


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if len(t) > 1 and t.lower() not in STOP]


def _extract(path: Path) -> str:
    """Extract text from PDF, Markdown, or plain text. PDFs require the pypdf package."""
    if path.suffix.lower() == '.pdf':
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return '\n\n'.join((page.extract_text() or '') for page in reader.pages)
        except ImportError:
            return ''
        except Exception:
            return ''
    try:
        return path.read_text(encoding='utf-8')
    except (OSError, UnicodeError):
        return ''


def _split_chunks(text: str):
    # Split first by paragraph, then by character windows to keep long PDF pages bounded.
    paragraphs = [re.sub(r'\s+', ' ', p).strip() for p in re.split(r'\n\s*\n|(?<=\.)\s{2,}', text) if p.strip()]
    chunks = []
    for paragraph in paragraphs:
        if len(paragraph) <= _CHUNK_SIZE:
            if len(paragraph) >= 25:
                chunks.append(paragraph)
            continue
        start = 0
        while start < len(paragraph):
            end = min(len(paragraph), start + _CHUNK_SIZE)
            if end < len(paragraph):
                boundary = paragraph.rfind(' ', start + _CHUNK_SIZE // 2, end)
                if boundary > start:
                    end = boundary
            piece = paragraph[start:end].strip()
            if len(piece) >= 25:
                chunks.append(piece)
            if end >= len(paragraph):
                break
            start = max(start + 1, end - _CHUNK_OVERLAP)
    return chunks


def _signature():
    rows = []
    if DOCS_DIR.exists():
        for p in sorted(DOCS_DIR.rglob('*')):
            if p.is_file() and p.suffix.lower() in {'.md', '.txt', '.pdf'}:
                try:
                    st = p.stat(); rows.append((str(p), st.st_mtime_ns, st.st_size))
                except OSError:
                    pass
    return tuple(rows)


def _build_index(force=False):
    global _cache_signature, _cache_chunks, _cache_df, _cache_idf
    sig = _signature()
    with _cache_lock:
        if not force and sig == _cache_signature:
            return _cache_chunks, _cache_df, _cache_idf
        chunks = []
        for path_str, _, _ in sig:
            path = Path(path_str)
            content = _extract(path)
            for n, chunk_text in enumerate(_split_chunks(content)):
                toks = Counter(_tokens(chunk_text))
                if toks:
                    chunks.append({'text': chunk_text, 'source': str(path.relative_to(DOCS_DIR)), 'chunk': n, 'tokens': toks})
        df = Counter()
        for c in chunks:
            df.update(c['tokens'].keys())
        total = max(1, len(chunks))
        idf = {t: math.log(1 + total / (1 + freq)) for t, freq in df.items()}
        _cache_signature, _cache_chunks, _cache_df, _cache_idf = sig, chunks, df, idf
        return chunks, df, idf


def ingest_documents(force: bool = False) -> dict:
    """Build/refresh the local index and report ingestion statistics."""
    chunks, _, _ = _build_index(force=force)
    sources = sorted({c['source'] for c in chunks})
    return {'ok': True, 'documents_indexed': len(sources), 'chunks_indexed': len(chunks), 'sources': sources,
            'supported_formats': ['.pdf', '.md', '.txt'], 'chunk_size': _CHUNK_SIZE, 'chunk_overlap': _CHUNK_OVERLAP}


def search_knowledge(query: str, top_k: int = 3) -> dict:
    """Retrieve ranked, source-labelled passages from the cached local index."""
    if not isinstance(query, str) or not query.strip():
        return {'ok': False, 'query': query, 'results': [], 'message': 'Provide a non-empty question.'}
    chunks, _, idf = _build_index()
    qtokens = Counter(_tokens(query))
    if not chunks or not qtokens:
        return {'ok': True, 'query': query, 'results': [], 'message': 'No matching store knowledge found. Do not guess; say the information is not available.'}
    scored = []
    qnorm = math.sqrt(sum(v*v for v in qtokens.values())) or 1.0
    for c in chunks:
        dot = sum(min(c['tokens'].get(t, 0), 3) * idf.get(t, 0) * min(qcount, 2) for t, qcount in qtokens.items())
        if dot:
            dnorm = math.sqrt(sum((min(tf, 3) * idf.get(t, 0))**2 for t, tf in c['tokens'].items())) or 1.0
            score = dot / (qnorm * dnorm)
            scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [{'source': c['source'], 'chunk': c['chunk'], 'text': c['text'], 'score': round(score, 4)}
               for score, c in scored[:max(1, min(int(top_k), 5))]]
    return {'ok': True, 'query': query, 'results': results,
            'message': 'Retrieved local store knowledge.' if results else 'No matching store knowledge found. Do not guess; say the information is not available.'}


def answer_from_knowledge(query: str, min_score: float = 0.12) -> dict | None:
    """Return a document-grounded answer for likely policy/FAQ questions only when retrieval is strong.

    This fast path avoids an LLM call for answers already stated in the approved store documents.
    None means the LLM/tool workflow should handle the turn.
    """
    q = query.lower()
    topics = ('shipping', 'delivery', 'return', 'refund', 'payment', 'pay', 'privacy', 'policy', 'card', 'otp',
              'courier', 'shipment', 'tracking', 'store policy', 'how long', 'payment gateway')
    if not any(t in q for t in topics):
        return None
    result = search_knowledge(query, top_k=5)
    rows = result.get('results') or []
    if not rows:
        return None

    # Prefer substantive policy passages over short section headings or demo disclaimers.
    # Small intent boosts help common natural-language questions select the relevant rule.
    ql = q.lower()
    ranked = []
    for row in rows:
        text = row.get('text', '')
        low = text.lower()
        if len(text.strip()) < 120 or 'demo notice:' in low or 'sample policy for demonstration and testing only' in low:
            continue
        score = float(row.get('score', 0.0))
        if any(term in ql for term in ('refund', 'money back', 'get my money')):
            if '5–7 business days' in low or '5-7 business days' in low or 'initiated within' in low:
                score += 0.25
            if 'refund' in low:
                score += 0.04
        if any(term in ql for term in ('return', 'exchange', 'send it back')):
            if '7 calendar days' in low:
                score += 0.20
            if 'return window' in low or 'return may be requested' in low:
                score += 0.08
        if any(term in ql for term in ('damaged', 'wrong item', 'incorrect item')) and '48 hours' in low:
            score += 0.25
        ranked.append((score, row))
    if not ranked:
        return None
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best = ranked[0]
    if float(best.get('score', 0.0)) < min_score and best_score < min_score + 0.08:
        return None
    # The retrieved passage is the response: no LLM is needed for this high-confidence fast path.
    return {'answer': best['text'], 'source': best['source'], 'score': round(best_score, 4), 'grounded': True}
