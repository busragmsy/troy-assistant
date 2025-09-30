import os
import psycopg2
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class EnhancedSearchResult:
    content_id: int
    title: str
    description: str
    content_type: str
    category: str
    difficulty_level: str
    score: float
    related_screens: List[str]
    tags: List[str]
    estimated_duration: Optional[int] = None

def _conn():
    return psycopg2.connect(
        host=os.environ.get('DB_HOST', 'pg'),
        port=int(os.environ.get('DB_PORT', '5432')),
        database=os.environ.get('DB_NAME', 'kb'),
        user=os.environ.get('DB_USER', 'troy'),
        password=os.environ.get('DB_PASSWORD', 'troy1234')
    )

def unified_search(query: str, limit: int = 20) -> List[EnhancedSearchResult]:
    conn = _conn(); cur = conn.cursor()
    try:      
        q = f"%{query}%"
        
        sql = """
        WITH unified_content AS (
          SELECT 
            d.id as content_id,
            d.title,
            COALESCE(ds.content, d.title) as description,
            'document' as content_type,
            d.department as category,
            'intermediate' as difficulty_level,
            ts_rank(
              to_tsvector('turkish', unaccent(d.title || ' ' || COALESCE(ds.content, ''))),
              plainto_tsquery('turkish', unaccent(%s))
            ) as score,
            ARRAY[]::text[] as related_screens,
            COALESCE(array_agg(DISTINCT dk.keyword), ARRAY[]::text[]) as tags,
            NULL::integer as estimated_duration
          FROM documents d
          LEFT JOIN document_sections ds ON d.id = ds.document_id
          LEFT JOIN document_keywords dk ON d.id = dk.document_id
          WHERE d.status = 'active'
            AND (
              d.title ILIKE %s OR ds.content ILIKE %s OR 
              EXISTS (SELECT 1 FROM document_keywords k WHERE k.document_id=d.id AND k.keyword ILIKE %s)
            )
          GROUP BY d.id, d.title, d.department, ds.content
          
          UNION ALL
          
          SELECT 
            tc.id as content_id,
            tc.title,
            tc.description,
            'training' as content_type,
            tc.topic_category as category,
            tc.difficulty_level,
            CASE 
              WHEN tc.title ILIKE %s THEN 3.0
              WHEN tc.description ILIKE %s THEN 2.0
              WHEN %s = ANY(tc.tags) THEN 1.5
              ELSE 1.0
            END as score,
            COALESCE(tc.related_screens, ARRAY[]::text[]) as related_screens,
            COALESCE(tc.tags, ARRAY[]::text[]) as tags,
            tc.estimated_duration_minutes as estimated_duration
          FROM training_content tc
          WHERE tc.status = 'active'
            AND (tc.title ILIKE %s OR tc.description ILIKE %s OR %s = ANY(tc.tags))
        )
        SELECT * FROM unified_content
        ORDER BY score DESC, content_type
        LIMIT %s
        """
        
        params = [query, q, q, q, q, q, query, q, q, query, limit]
        cur.execute(sql, params)
        rows = cur.fetchall()
        
        res = []
        for r in rows:
            res.append(EnhancedSearchResult(
                content_id=r[0],
                title=r[1],
                description=(r[2][:300] + '...') if r[2] and len(r[2]) > 300 else (r[2] or ''),
                content_type=r[3],
                category=r[4] or '',
                difficulty_level=r[5] or 'beginner',
                score=float(r[6] or 0.0),
                related_screens=r[7] or [],
                tags=r[8] or [],
                estimated_duration=r[9]
            ))
        return res
    finally:
        cur.close(); conn.close()