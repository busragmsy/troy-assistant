# kb/ingest/emails_txt.py
import os, re, argparse, psycopg2

TOPIC_KEYWORDS = {
    'fiyat': ['fiyat','revize','marj','psf','markup'],
    'indirim': ['indirim','kampanya','discount','promosyon'],
    'stok': ['stok','inventory','depo','mağaza','magaza'],
    'rapor': ['rapor','report','analiz','dashboard'],
    'troy_screens': ['troy','ekran','menü','menu','navigation'],
}
LEVEL_KEYWORDS = {
    'beginner': ['temel','nasıl','nasil','adım adım','adim adim','giriş','giris'],
    'intermediate': ['detay','gelişmiş','gelismis','özelleştirme','ozellestirme','konfigürasyon','konfigurasyon'],
    'advanced': ['entegrasyon','otomasyon','batch','toplu'],
}
TAGS_TERMS = ['troy','fiyat','revize','indirim','stok','rapor','ekran','işlem','islem','onay','talep',
              'liste','müşteri','musteri','ürün','urun','kategori','sezon','mağaza','magaza']

def connect():
    return psycopg2.connect(
        host=os.getenv('DB_HOST','pg'),
        port=int(os.getenv('DB_PORT','5432')),
        dbname=os.getenv('DB_NAME','kb'),
        user=os.getenv('DB_USER','troy'),
        password=os.getenv('DB_PASSWORD','troy1234'),
    )

def classify(text:str):
    t=text.lower()
    topic='general'
    for k,ws in TOPIC_KEYWORDS.items():
        if any(w in t for w in ws): topic=k; break
    level='beginner'
    for k,ws in LEVEL_KEYWORDS.items():
        if any(w in t for w in ws): level=k; break
    return topic, level

STEP_PATTERNS = [
    r'^\s*\d+[\.\)]\s+(.*\S)',                 # 1. Adım  /  1) Adım
    r'^\s*(?:-|\*|•|·|–|—)\s+(.*\S)',          # - Adım   • Adım
    r'^\s*[a-z]\)\s+(.*\S)',                   # a) Adım
    r'^\s*[ivxlcdm]+\)\s+(.*\S)',              # i) ii)  roman
]
IMPERATIVE_STARTS = [
    'aç', 'gir', 'seç', 'tıkla', 'kaydet', 'gönder',
    'kontrol et', 'atan', 'oluştur', 'doldur', 'filtrele',
    'ara', 'sil', 'ekle', 'güncelle', 'onayla', 'reddet',
    'başlat', 'bitir', 'yazdır'
]

def extract_steps(text: str):
    # 1) Madde işaretlerini yakala
    for pat in STEP_PATTERNS:
        matches = [m.group(1).strip() for m in re.finditer(pat, text, flags=re.MULTILINE | re.IGNORECASE)]
        if matches:
            return matches[:20]
    steps = []
    for ln in text.splitlines():
        s = ln.strip()
        if len(s) < 5:
            continue
        lw = s.lower()
        if any(lw.startswith(v) for v in IMPERATIVE_STARTS):
            steps.append(s)
    return steps[:20]

def split_title_body(path:str):
    for enc in ('utf-8','cp1254'):
        try:
            with open(path,'r',encoding=enc) as f:
                lines=[ln.rstrip() for ln in f]
            break
        except UnicodeDecodeError:
            continue
    if not lines:
        base=os.path.splitext(os.path.basename(path))[0]
        return base[:500],''
    title=(lines[0] or os.path.basename(path))[:500]
    body='\n'.join(lines[1:]).strip() if len(lines)>1 else lines[0].strip()
    return title, body

def make_tags(text:str):
    t=text.lower()
    return [kw for kw in TAGS_TERMS if kw in t][:10]

def upsert_training(cur, title, desc, monthly_theme, campaign_id, steps, tags, update_if_exists=False):
    topic, level = classify(title+' '+desc)
    # dupe önleme
    cur.execute("SELECT id FROM training_content WHERE email_campaign_id=%s AND title=%s LIMIT 1",
                (campaign_id, title))
    row=cur.fetchone()
    if row:
        if update_if_exists:
            cur.execute("""
                UPDATE training_content
                SET description=%s, step_by_step=%s, tags=%s, monthly_theme=%s, updated_at=NOW()
                WHERE id=%s
                RETURNING id
            """, (desc, steps, tags, monthly_theme, row[0]))
            return row[0], 'updated'
        return row[0], 'skipped'

    cur.execute("""
        INSERT INTO training_content
        (title, description, content_type, topic_category, difficulty_level,
         thumbnail_url, step_by_step, related_screens, tags, monthly_theme,
         email_campaign_id, status)
        VALUES (%s, %s, 'tutorial', %s, %s,
                NULL, %s, ARRAY[]::text[], %s, %s,
                %s, 'active')
        RETURNING id
    """, (title, desc, topic, level, steps, tags, monthly_theme, campaign_id))
    return cur.fetchone()[0], 'inserted'

def process_txt_dir(root_dir:str, campaign_id:str, monthly_theme:str):
    conn=connect(); cur=conn.cursor()
    inserted=skipped=failed=0
    try:
        for base,_,files in os.walk(root_dir):
            for fn in sorted(files):
                if not fn.lower().endswith('.txt'): continue
                path=os.path.join(base,fn)
                try:
                    title,body=split_title_body(path)
                    steps=extract_steps(body)
                    tags=make_tags(title+' '+body)
                    tid,st=upsert_training(cur,title,body,monthly_theme,campaign_id,steps,tags, update_if_exists=True)
                    if st=='inserted':
                        inserted+=1; print(f"✅ {fn} -> training_id={tid}")
                    else:
                        skipped+=1; print(f"⏭️  {fn} -> already exists (training_id={tid})")
                except Exception as e:
                    failed+=1; print(f"❌ {fn} -> {e}")
        conn.commit()
    finally:
        cur.close(); conn.close()
    print(f"\nSummary: inserted={inserted}, skipped={skipped}, failed={failed}")

def main():
    ap=argparse.ArgumentParser(description="TXT e-posta içeriklerini training_content tablosuna yükler.")
    ap.add_argument('--dir', required=True)
    ap.add_argument('--campaign-id', required=True)
    ap.add_argument('--monthly-theme', default='Genel Bilgilendirme')
    args=ap.parse_args()
    if not os.path.isdir(args.dir): raise SystemExit(f"Dizin bulunamadı: {args.dir}")
    process_txt_dir(args.dir, args.campaign_id, args.monthly_theme)

if __name__=='__main__': main()
