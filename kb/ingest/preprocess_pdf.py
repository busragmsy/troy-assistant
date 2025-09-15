# kb/ingest/preprocess_pdf.py
from pypdf import PdfReader
import regex as re
from unidecode import unidecode

SECTION_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\)|\d+(?:\.\d+)*\s+|[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ\s]{5,}|•|-)\s+",
    re.MULTILINE
)

def clean_text(t: str) -> str:
    # Temizlik: fazla boşluk, kırık satırlar vs.
    t = t.replace('\r', '')
    # Peşpeşe 3+ boş satırı 2'ye düşür
    t = re.sub(r'\n{3,}', '\n\n', t)
    # Satır sonu tire kopmaları: "fiyat-\nlandırma" -> "fiyatlandırma"
    t = re.sub(r'-\n', '', t)
    # Çok boşluk -> tek boşluk (satır içi)
    t = re.sub(r'[ \t]{2,}', ' ', t)
    return t.strip()

def extract_pages(pdf_path: str):
    reader = PdfReader(pdf_path)
    pages = []
    for i, p in enumerate(reader.pages, start=1):
        try:
            txt = p.extract_text() or ""
        except Exception:
            txt = ""
        pages.append((i, clean_text(txt)))
    return pages

def split_sections(pages):
    """
    Sayfa metinlerini birleştir, SECTION_RE ile bölümlere ayır, her bölümün
    başlangıç ve bitiş sayfa numarasını koru.
    """
    # sayfa sınırlarını koruyarak tek metin inşa et
    offsets = []  # [(start_char, end_char, page_no)]
    buf = []
    pos = 0
    for num, txt in pages:
        start = pos
        buf.append(txt + "\n\n")  # sayfalar arası boşluk
        pos += len(txt) + 2
        offsets.append((start, pos, num))
    full = "".join(buf)

    # Bölümleri işaretle
    sections = []
    last_idx = 0
    for m in SECTION_RE.finditer(full):
        start = m.start()
        if start > last_idx:
            sections.append((last_idx, start))
        last_idx = start
    # son parça
    if last_idx < len(full):
        sections.append((last_idx, len(full)))

    # bölüm -> sayfa aralığı çöz
    def span_to_pages(s, e):
        ps = []
        for (a, b, pg) in offsets:
            if b <= s:  # bu sayfa tamamen önce
                continue
            if a >= e:  # bu sayfa tamamen sonra
                break
            ps.append(pg)
        if not ps:
            return (None, None)
        return (min(ps), max(ps))

    enriched = []
    for (s, e) in sections:
        text = full[s:e].strip()
        if not text:
            continue
        p1, p2 = span_to_pages(s, e)
        enriched.append({
            "text": text,
            "page_start": p1,
            "page_end": p2,
        })
    return enriched

def chunkify_section(text: str, max_tokens: int = 1000, overlap: int = 150):
    """
    Basit kelime sayacı ile chunking; istersen tiktoken kullanabilirsin.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        j = min(i + max_tokens, len(words))
        chunk = " ".join(words[i:j])
        chunks.append(chunk.strip())
        if j == len(words):
            break
        i = max(0, j - overlap)
    return chunks

def detect_section_title(chunk_text: str):
    # ilk satırı veya ilk kalın başlık havasındaki kısmı başlık say
    first_line = chunk_text.splitlines()[0].strip()
    # çok uzun ise kısalt
    return (first_line[:120] + "…") if len(first_line) > 120 else first_line

def parse_pdf_advanced(pdf_path: str, lang: str = "tr", max_tokens=1000, overlap=150):
    pages = extract_pages(pdf_path)
    sections = split_sections(pages)
    results = []
    for sec in sections:
        sec_text = sec["text"]
        p1, p2 = sec["page_start"], sec["page_end"]
        chunks = chunkify_section(sec_text, max_tokens=max_tokens, overlap=overlap)
        for ch in chunks:
            title = detect_section_title(ch)
            results.append({
                "text": ch,
                "section": title,
                "page_start": p1,
                "page_end": p2,
                "lang": lang,
                "token_count": len(ch.split())
            })
    return results
