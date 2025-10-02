import os
import re
import json
import hashlib
from typing import List, Dict, Tuple
from pypdf import PdfReader

# =========================
# 1) Konfigürasyon
# =========================
PDF_DIR = r"C:\troy-assistant\kb\data\docs"     # Giriş klasörü
OUTPUT_FILE = "temiz_rag_chunks.jsonl"          # Çıkış JSONL
ENCODING = "utf-8"

# Token ~ karakter yaklaşık dönüşümü (4 char ≈ 1 token)
CHARS_PER_TOKEN = 4

# İç chunklama (başlık altı parçalara bölme) için hedef uzunluk/örtüşme
TARGET_TOKENS = 700
OVERLAP_TOKENS = 120

TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN

# =========================
# 2) Desenler (Regex)
# =========================
# (A) Header/Footer & Gürültü
HEADER_FOOTER_PATTERN = re.compile(
    r"(KULLANICI KILAVUZU|TALİMAT).*?(DAHİLİ|GENEL)", re.IGNORECASE | re.DOTALL
)

# Satır bazlı "sayfa 5", "5/26" veya tek başına sayılar (maks 3-4 karakter) gibi numaraları silecek,
# ama gövdedeki ciddi sayıları dokunmadan bırakacak bir yaklaşım:
LINE_PAGE_NUMBER_PATTERN = re.compile(
    r"^\s*(sayfa\s*\d+|\d+\s*/\s*\d+|\d{1,3})\s*$", re.IGNORECASE | re.MULTILINE
)

VISUAL_REF_PATTERN = re.compile(r"\b(Görsel|Tablo|Şekil)[- ]?\d+\b", re.IGNORECASE)

# (B) Bölüm Başlıkları (hiyerarşik)
CHUNK_START_PATTERN = re.compile(
    r"(?m)^\s*(\d+(?:\.\d+)*)(?:\.|\))?\s+([A-ZÇĞİÖŞÜ].+?)\s*$"
)

# (C) İmza blokları / roller (çok tekrar eden kuyruk metinleri)
SIGNOFF_PATTERN = re.compile(
    r"(Hazırlayan|Kontrol Eden|Onaylayan|OPERASYON MÜDÜRLÜĞÜ|FİYAT VE İNDİRİM PLANLAMA MÜDÜRÜ).*",
    re.IGNORECASE
)

# İçindekiler (TOC) blok temizliği: belgelerin başında yer alan kısım
TOC_PATTERN = re.compile(
    r"(?is)\bİÇİNDEKİLER\b.*?(?:\n\s*\d+(\.\d+)*\s+|(?:\n){2,})"
)


# =========================
# 3) Yardımcılar
# =========================
def approx_token_count(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)

def hash_id(*parts: str) -> str:
    h = hashlib.sha1(("||".join(parts)).encode("utf-8")).hexdigest()
    return h[:16]

def normalize_spaces(text: str) -> str:
    # fazla boşlukları ve boş satırları sadeleştir
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_page_text(text: str) -> str:
    if not text:
        return ""
    # İçindekiler sayfasındaki başlığı sayfa bazında gelirse zayıf dokunuş
    text = HEADER_FOOTER_PATTERN.sub(" ", text)
    text = LINE_PAGE_NUMBER_PATTERN.sub("", text)
    text = VISUAL_REF_PATTERN.sub(" ", text)
    text = SIGNOFF_PATTERN.sub("", text)
    # Kaynakta bazen satır sonlarında kırık tireler bulunabilir: "fiyatla-\nrma"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Normalizasyon
    text = normalize_spaces(text)
    return text

def remove_toc_block(full_text: str) -> str:
    # Belgenin başı yakınlarında ise temizle
    head = full_text[:8000]
    match = TOC_PATTERN.search(head)
    if match:
        start, end = match.span()
        return full_text[:start] + full_text[end:]
    return full_text

def paged_extract(reader: PdfReader) -> List[str]:
    pages = []
    for p in reader.pages:
        try:
            raw = p.extract_text() or ""
        except Exception:
            raw = ""
        pages.append(raw)
    return pages

def map_index_to_page(index: int, page_offsets: List[int]) -> int:
    # page_offsets: her sayfanın birleşik metin içindeki başlangıç indeksi
    # index hangi sayfaya düşüyor?
    lo, hi = 0, len(page_offsets) - 1
    ans = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if page_offsets[mid] <= index:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ans + 1  # 1-based page number

def sliding_window_chunks(text: str, max_chars: int, overlap_chars: int) -> List[Tuple[int, int]]:
    """Başlık altı büyük içerikleri hedef uzunlukta parçala."""
    spans = []
    n = len(text)
    if n == 0:
        return spans
    if n <= max_chars:
        return [(0, n)]
    start = 0
    while start < n:
        end = min(n, start + max_chars)
        # Parçayı kelime sınırında bitirmeye çalış
        if end < n:
            m = re.search(r"\s", text[end:end+400])
            if m:
                end = end + m.start()
        spans.append((start, end))
        if end >= n:
            break
        start = max(0, end - overlap_chars)
    return spans

def sectionize_with_titles(text: str) -> List[Tuple[str, int, int]]:
    """
    Başlıkları yakalayıp (title, start_idx, end_idx) döndürür.
    end_idx: bir sonraki başlığın start'ına kadar.
    Başlık bulunamazsa tek bir 'Doküman Genel İçeriği' bölümü döner.
    """
    sections = []
    matches = list(CHUNK_START_PATTERN.finditer(text))
    if not matches:
        return [("Doküman Genel İçeriği", 0, len(text))]
    for i, m in enumerate(matches):
        title = (m.group(1) + " " + m.group(2)).strip()
        s = m.end()
        e = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        if s < e:
            sections.append((title, s, e))
    return sections


# =========================
# 4) PDF İşleme
# =========================
def process_pdf(file_path: str) -> List[Dict]:
    file_name = os.path.basename(file_path)
    print(f"-> {file_name} işleniyor...")

    try:
        reader = PdfReader(file_path)
    except Exception as e:
        print(f"  HATA: {file_name} açılırken: {e}")
        return []

    raw_pages = paged_extract(reader)
    if not any(raw_pages):
        print(f"  UYARI: {file_name} içeriği boş görünüyor.")
        return []

    # Sayfa bazlı temizlik
    cleaned_pages = [clean_page_text(p) for p in raw_pages]

    # Belge başındaki İÇİNDEKİLER'i belge genelinde kaldır (sayfa bazında kaçmış olabilir)
    joined_for_toc = "\n\n".join(cleaned_pages)
    joined_for_toc = remove_toc_block(joined_for_toc)

    # Sayfa offsetlerini yeniden hesapla
    # Sayfaları tekrar böl: "\n\n[[PAGE_BREAK]]\n\n" ile
    PAGE_SEP = "\n\n[[PAGE_BREAK]]\n\n"
    # joined_for_toc'u tekrar sayfa listesine bölmek hassas olabilir; daha güvenlisi:
    # temizlenmiş sayfaları tek tek birleştirip offset tut.
    concat_text = ""
    page_offsets = []
    for idx, page_text in enumerate(cleaned_pages):
        page_offsets.append(len(concat_text))
        concat_text += page_text
        if idx < len(cleaned_pages) - 1:
            concat_text += PAGE_SEP

    # concat_text içinde İÇİNDEKİLER temizliğini de uygulayalım:
    concat_text = remove_toc_block(concat_text)
    concat_text = normalize_spaces(concat_text)

    # Bölümlere ayır ve içeride sliding window uygula
    sections = sectionize_with_titles(concat_text)

    chunks_out = []
    running_index = 0

    for sec_idx, (section_title, s, e) in enumerate(sections):
        section_text = concat_text[s:e].strip()
        if not section_text:
            continue

        spans = sliding_window_chunks(section_text, TARGET_CHARS, OVERLAP_CHARS)
        for i, (a, b) in enumerate(spans):
            sub = section_text[a:b].strip()
            if not sub:
                continue

            # absolute indeksler (concat_text içindeki)
            abs_start = s + a
            abs_end = s + b

            # sayfa aralığını hesapla
            p_start = map_index_to_page(abs_start, page_offsets)
            p_end = map_index_to_page(abs_end, page_offsets)

            # kimlik
            chunk_index = f"{sec_idx:03d}-{i:03d}"
            cid = hash_id(file_name, section_title, str(p_start), str(p_end), chunk_index, sub[:64])

            chunks_out.append({
                "chunk_id": cid,
                "file_name": file_name,
                "section_title": section_title,
                "content": sub,
                "page_start": p_start,
                "page_end": p_end,
                "chunk_index": chunk_index,
                "approx_tokens": approx_token_count(sub)
            })

    print(f"   -> {len(chunks_out)} adet chunk üretildi.")
    return chunks_out


# =========================
# 5) Ana Döngü
# =========================
def main():
    if not os.path.exists(PDF_DIR):
        print(f"HATA: Klasör bulunamadı: {PDF_DIR}")
        return

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"HATA: '{PDF_DIR}' içinde PDF yok.")
        return

    all_chunks: List[Dict] = []

    for pdf in pdf_files:
        path = os.path.join(PDF_DIR, pdf)
        chunks = process_pdf(path)
        all_chunks.extend(chunks)

    with open(OUTPUT_FILE, "w", encoding=ENCODING) as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("--- İŞLEM TAMAMLANDI ---")
    print(f"Toplam chunk: {len(all_chunks)}")
    print(f"Çıktı: {os.path.abspath(OUTPUT_FILE)}")
    print("=" * 60)
    print("\nSonraki adımlar:")
    print("1) Embedding: 'content' için vektör üret, metadata'yı (file_name, section_title, page_start/end, chunk_id) koru.")
    print("2) DB yükleme: (pgvector/Weaviate/FAISS) -> embedding + metadata yaz.")
    print("3) Sorgu zamanı: metadata filtreleri (file_name, sayfa aralığı, section_title) ile sonuç kalitesini artır.")


if __name__ == "__main__":
    main()
