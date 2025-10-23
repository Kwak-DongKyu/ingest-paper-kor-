# ingest_papers.py
# ============================================================
# PDF -> GPT(메타/본문/한줄요약) -> Notion DB 기록
# + Notion file_uploads REST로 PDF/이미지 직접 업로드
# + 논문 URL Bookmark + (섞기 레이아웃) Sections & Figures + Contents
# + Notion DB 속성 자동 매핑/보강 (중복 매핑 금지, 실패시 안전 생성)
# ============================================================

# 0) Imports
import os, re, glob, json, sys, io, mimetypes
from urllib.parse import quote_plus
import requests
from dotenv import load_dotenv
from pypdf import PdfReader
from notion_client import Client as NotionClient
from notion_client.errors import APIResponseError
from openai import OpenAI

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# ── 설정
PROMOTE_FIRST_FIGURE = True
SCALE, MARGIN, MAX_VGAP = 2.0, 6.0, 120.0
MIN_X_OVERLAP_RATIO, MERGE_NEAR_IMAGES, SAVE_LOCAL_FIGS = 0.20, True, False
CAPTION_RE = re.compile(r"^\s*(?:(?:Figure|Fig\.?)\s*[0-9IVX]+|(?:그림|도)\s*\d+)\b", re.IGNORECASE)
SECTION_KEYS = [
    "1. 기존문제", "2. 선행연구", "3. 이번 연구의 개선점",
    "4. 문제의 중요성", "5. 제안 시스템/방법", "6. 실험 가설/절차",
]

# ── ENV/Clients
load_dotenv()
NOTION_TOKEN   = os.getenv("NOTION_TOKEN")
DATABASE_ID    = os.getenv("NOTION_DATABASE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "chatgpt-4o-latest")
NOTION_VERSION = "2022-06-28"
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "700"))
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

if not (NOTION_TOKEN and DATABASE_ID and OPENAI_API_KEY):
    print("❌ .env에 NOTION_TOKEN / NOTION_DATABASE_ID / OPENAI_API_KEY가 필요합니다.")
    sys.exit(1)

notion = NotionClient(auth=NOTION_TOKEN)
oai    = OpenAI(api_key=OPENAI_API_KEY)

# ── DB 스키마
REQUIRED_SCHEMA = {
    "Title":                 {"type": "title"},
    "Tag":                   {"type": "multi_select"},
    "Authors":               {"type": "rich_text"},
    "Year":                  {"type": "rich_text"},
    "Conference/journal":    {"type": "rich_text"},
    "One sentence":          {"type": "rich_text"},
}

ALIASES = {
    "Title": ["title", "name", "paper title", "논문 제목"],
    "Tag": ["tag", "tags", "토픽", "키워드"],
    "Authors": ["authors", "author", "저자", "저자명"],
    "Year": ["year", "연도", "발표연도"],
    "Conference/journal": ["conference/journal", "venue", "conference", "journal", "학회", "저널"],
    "One sentence": ["one sentence", "summary", "short summary", "한줄요약", "요약"],
}

def s(v): return "" if v is None else str(v)

def strip_unpaired_surrogates(text: str) -> str:
    if not isinstance(text, str): return text
    text = _SURROGATE_RE.sub("", text)
    try:
        text = text.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")
    except Exception:
        text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return text.replace("\x00", "")

# ── DB 매핑 (중복 금지 + 실패 시 안전 생성)
def _norm(s: str) -> str:
    return (s or "").strip().lower()

def _find_title_prop_key(db: dict) -> str | None:
    for k, v in db.get("properties", {}).items():
        if v.get("type") == "title":
            return k
    return None

def _match_by_alias(db: dict, want_name: str, want_type: str, used: set[str]) -> str | None:
    """타입이 want_type이고, 이름이 별칭과 유사한 미사용 키를 찾는다."""
    props = db.get("properties", {})
    aliases = [_norm(want_name)] + [_norm(a) for a in ALIASES.get(want_name, [])]
    cands = [k for k, v in props.items() if v.get("type") == want_type and k not in used]

    for k in cands:
        if _norm(k) in aliases:
            return k
    for k in cands:
        nk = _norm(k).replace(" ", "").replace("-", "").replace("_", "")
        for a in aliases:
            na = a.replace(" ", "").replace("-", "").replace("_", "")
            if na and na in nk:
                return k
    return None

def ensure_db_property_map(dbid: str) -> dict:
    """
    1) 현재 DB 조회
    2) 없는 속성들 '모두' 모아서 한 번의 update로 생성
    3) 최신 DB를 다시 조회해 최종 매핑 반환
    """
    db = notion.databases.retrieve(dbid)
    props = db.get("properties", {})
    used: set[str] = set()
    mapping: dict[str, str] = {}

    # 1) Title(필수)
    title_key = _find_title_prop_key(db)
    if not title_key:
        # 타이틀이 없으면 먼저 타이틀만 생성
        notion.databases.update(database_id=dbid, properties={"Title": {"title": {}}})
        db = notion.databases.retrieve(dbid)
        props = db.get("properties", {})
        title_key = _find_title_prop_key(db)
    mapping["Title"] = title_key
    used.add(title_key)

    # 2) 나머지 원하는 속성 중 이미 있는 것 매칭
    missing_props: dict[str, dict] = {}
    for want_name, meta in REQUIRED_SCHEMA.items():
        if want_name == "Title":
            continue
        want_type = meta["type"]

        # 타입/별칭으로 매칭
        found = _match_by_alias(db, want_name, want_type, used)
        if found:
            mapping[want_name] = found
            used.add(found)
        else:
            # 아직 없으니 "한꺼번에" 만들 목록에 추가
            missing_props[want_name] = {want_type: {}}

    # 3) 부족한 속성들 한 번의 update로 생성 (있을 때만)
    if missing_props:
        notion.databases.update(database_id=dbid, properties=missing_props)
        # 생성 직후 최신 DB 재조회
        db = notion.databases.retrieve(dbid)
        props = db.get("properties", {})

        # 생성된 이름 그대로 매핑
        for want_name in missing_props.keys():
            if want_name in props:
                mapping[want_name] = want_name
                used.add(want_name)
            else:
                # 이름 충돌 등으로 생성이 다른 이름으로 되었을 가능성 → 타입/별칭으로 재검색
                found = _match_by_alias(db, want_name, REQUIRED_SCHEMA[want_name]["type"], used)
                if found:
                    mapping[want_name] = found
                    used.add(found)
                else:
                    # 최후 폴백: 원하는 이름으로 기록(실패 로그)
                    print(f"⚠️ 속성 생성/매칭 실패: {want_name}")
                    mapping[want_name] = want_name

    return mapping

# ── Notion upload helpers
def _notion_headers(extra=None):
    base = {"Authorization": f"Bearer {NOTION_TOKEN}", "Notion-Version": NOTION_VERSION}
    if extra: base.update(extra)
    return base

def create_file_upload(filename: str, content_type: str) -> dict:
    url = "https://api.notion.com/v1/file_uploads"
    payload = {"filename": filename, "content_type": content_type}
    r = requests.post(url, headers=_notion_headers({"Content-Type":"application/json"}), data=json.dumps(payload), timeout=30)
    r.raise_for_status(); return r.json()

def send_file_upload_bytes(file_upload_id: str, data: bytes, filename: str, content_type: str) -> dict:
    url = f"https://api.notion.com/v1/file_uploads/{file_upload_id}/send"
    files = {"file": (filename, io.BytesIO(data), content_type)}
    r = requests.post(url, headers=_notion_headers(), files=files, timeout=120)
    r.raise_for_status(); return r.json()

def send_file_upload_path(file_upload_id: str, file_path: str, content_type: str) -> dict:
    url = f"https://api.notion.com/v1/file_uploads/{file_upload_id}/send"
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, content_type)}
        r = requests.post(url, headers=_notion_headers(), files=files, timeout=120)
    r.raise_for_status(); return r.json()

def file_block_from_upload(file_upload_id: str) -> dict:
    return {"object":"block","type":"file","file":{"type":"file_upload","file_upload":{"id":file_upload_id}}}

def image_block_from_upload(file_upload_id: str, caption: str = "") -> dict:
    block = {"object":"block","type":"image","image":{"type":"file_upload","file_upload":{"id":file_upload_id}}}
    if caption: block["image"]["caption"] = [{"type":"text","text":{"content":caption}}]
    return block

# ── 기타 유틸/텍스트/마크다운
def guess_mime(path: str, default="application/octet-stream"):
    mime, _ = mimetypes.guess_type(path); return mime or default

def read_pdf(path: str, max_chars: int = 40000) -> str:
    try:
        reader = PdfReader(path); texts=[]
        for page in reader.pages:
            chunk = page.extract_text() or ""
            if chunk: texts.append(chunk)
            if sum(len(t) for t in texts) >= max_chars: break
        text = "\n\n".join(texts)
        text = re.sub(r"[ \t]+"," ", text); text = re.sub(r"\n{3,}","\n\n", text).strip()
        return text[:max_chars]
    except Exception as e:
        return f"[PDF READ ERROR] {e}"

def safe_year_guess(filename: str, text: str) -> str:
    cand = re.findall(r"\b(19[7-9]\d|20[0-4]\d|2025|2026)\b", filename + " " + text)
    return cand[0] if cand else ""

MD_PATTERNS = [
    (re.compile(r"\*\*(.+?)\*\*"), {"bold": True}),
    (re.compile(r"__(.+?)__"),     {"underline": True}),
    (re.compile(r"\*(.+?)\*"),     {"italic": True}),
]
def md_to_rich_text(md: str) -> list[dict]:
    spans=[(0, md, {})]
    for pat, anno in MD_PATTERNS:
        new=[]
        for _, text, attrs in spans:
            pos=0
            for m in pat.finditer(text):
                if m.start()>pos: new.append((0, text[pos:m.start()], attrs))
                new.append((0, m.group(1), {**attrs, **anno})); pos=m.end()
            if pos<len(text): new.append((0, text[pos:], attrs))
        spans=new
    rich=[]
    for _, piece, attrs in spans:
        if not piece: continue
        rich.append({"type":"text","text":{"content":piece},"annotations":{
            "bold":attrs.get("bold",False),"italic":attrs.get("italic",False),
            "underline":attrs.get("underline",False),"code":False,"strikethrough":False,"color":"default"}})
    return rich

# ── 외부 URL 찾기
def find_paper_url(title: str, authors: str = "", year: str = "") -> str:
    if not title: return ""
    title_q = quote_plus(title.strip())
    try:
        url = f"https://api.crossref.org/works?query.title={title_q}&rows=3"
        r = requests.get(url, timeout=10)
        if r.ok:
            items = r.json().get("message", {}).get("items", [])
            title_norm = re.sub(r"\s+"," ",title).strip().lower()
            best=None
            for it in items:
                it_title=" ".join(it.get("title", [])).strip()
                it_norm=re.sub(r"\s+"," ",it_title).strip().lower()
                score = 2 if it_norm==title_norm else (1 if title_norm in it_norm or it_norm in title_norm else 0)
                if year and str(it.get("issued",{}).get("date-parts",[[None]])[0][0])==year: score+=1
                if not best or score>best[0]: best=(score,it)
            if best and best[0]>=1:
                it=best[1]; doi=it.get("DOI"); link=it.get("URL")
                if doi: return f"https://doi.org/{doi}"
                if link: return link
    except Exception: pass
    try:
        r = requests.get(f"http://export.arxiv.org/api/query?search_query=ti:{title_q}", timeout=10)
        if r.ok and "<entry>" in r.text:
            m=re.search(r"<id>(https?://arxiv\.org/abs/[^\<]+)</id>", r.text)
            if m: return m.group(1).strip()
    except Exception: pass
    try:
        s2=f"https://api.semanticscholar.org/graph/v1/paper/search?query={title_q}&limit=1&fields=url,externalIds,title,year"
        r=requests.get(s2, timeout=10)
        if r.ok and r.json().get("data"):
            d0=r.json()["data"][0]
            if d0.get("url"): return d0["url"]
            ext=d0.get("externalIds",{}); 
            if "DOI" in ext: return f"https://doi.org/{ext['DOI']}"
    except Exception: pass
    return ""

# ── GPT Calls
def call_gpt_meta(text_snippet: str) -> dict:
    text_snippet = strip_unpaired_surrogates(text_snippet)
    system_msg = "너는 논문 메타데이터 추출에 특화된 비서다. 반드시 유효한 JSON만 반환해라."
    user_msg = f"""다음 논문 스니펫을 바탕으로 아래 키만 포함한 JSON을 만들어라.
- title
- authors
- year
- conference_journal
- tag

tag 생성 규칙:
1) 스니펫의 "CCS Concepts" 또는 유사 섹션에서 1~6개 핵심 키워드를 추출(없으면 추정).
2) 논문 분류용 태그 2~3개 추가(중복 금지).
3) 결과: 평탄 배열의 tag. 텍스트만.

스니펫:
\"\"\"{text_snippet}\"\"\""""
    resp = oai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
        temperature=0.2, response_format={"type":"json_object"}, max_tokens=OPENAI_MAX_TOKENS)
    raw = resp.choices[0].message.content
    try: return json.loads(raw)
    except Exception: return {"title":"","authors":"","year":"","conference_journal":"","tag":[],"_raw":raw}

def call_gpt_contents_marked(text_snippet: str) -> dict:
    text_snippet = strip_unpaired_surrogates(text_snippet)
    system_msg = "너는 논문 요약에 특화된 한국어 연구 비서다. 반드시 유효한 JSON만 반환해라."
    user_msg = f"""
다음 논문 스니펫을 바탕으로 'contents' 딕셔너리를 만들어라.
키는 아래 6개를 정확히 사용(순서/철자 유지):
- "1. 기존문제"
- "2. 선행연구"
- "3. 이번 연구의 개선점"
- "4. 문제의 중요성"
- "5. 제안 시스템/방법"
- "6. 실험 가설/절차"

규칙:
- 각 값은 한국어 단락. 첫 문장은 '키 이름 : '로 시작. (예: "1. 기존문제 : ...")
- **굵게**, __밑줄__, *이탤릭* 등 간단 마크다운 사용 OK.
- 핵심 개념에는 강조를 적극 사용.

스니펫:
\"\"\"{text_snippet}\"\"\""""
    resp = oai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
        temperature=0.2, response_format={"type":"json_object"}, max_tokens=2000)
    raw = resp.choices[0].message.content
    data = json.loads(raw)
    contents = data.get("contents", data)
    for k in SECTION_KEYS: contents.setdefault(k, "")
    return contents

def call_gpt_one_sentence(text_snippet: str) -> str:
    text_snippet = strip_unpaired_surrogates(text_snippet)
    resp = oai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":"한 문장(30자 이내)만 반환하라."},
                  {"role":"user","content":f"다음 논문을 한글로 30자 이내 한 문장 요약:\n\"\"\"{text_snippet}\"\"\""}],
        temperature=0.2, max_tokens=64)
    return (resp.choices[0].message.content or "").strip().replace("\n"," ")

def call_gpt_layout(contents: dict, figures: list[dict]) -> list[dict]:
    def _brief(s: str, n: int = 900) -> str: return strip_unpaired_surrogates((s or "")[:n])
    contents_brief = {k: _brief(contents.get(k, "")) for k in SECTION_KEYS}
    fig_brief = [{"id": f.get("id"), "page": int(f.get("page", 0) or 0), "caption": _brief(f.get("caption_text",""), 400)} for f in figures]

    system_msg = "너는 논문 편집 레이아웃 어시스턴트다. 반드시 유효한 JSON만 반환하라."
    user_msg = f"""
아래 '섹션 요약'과 '피겨 목록'을 바탕으로, 독자가 읽기 자연스러운 레이아웃을 만들어라.
출력은 반드시 {{"items":[ ... ]}} 형태의 JSON 객체.

규칙:
1) 섹션 6개를 이 순서로 모두 포함: {SECTION_KEYS}
2) 피겨는 관련성 높은 섹션 '바로 뒤'에 분산 배치. 마지막에 몰아넣지 말 것.
3) 아이템 형식:
   - 섹션: {{"type":"section","key":"<섹션키>"}}
   - 피겨:  {{"type":"figure","id":"<피겨 id>"}}
4) 같은 피겨 id는 한 번만 사용.

섹션 요약:
{json.dumps(contents_brief, ensure_ascii=False, indent=2)}

피겨 목록:
{json.dumps(fig_brief, ensure_ascii=False, indent=2)}
"""
    resp = oai.chat_completions.create if hasattr(oai, "chat_completions") else oai.chat.completions.create
    out = resp(model=OPENAI_MODEL,
               messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
               temperature=0.2, response_format={"type":"json_object"}, max_tokens=1200)
    raw = out.choices[0].message.content
    try:
        data = json.loads(raw); items = data.get("items", [])
        if not isinstance(items, list): raise ValueError("items not a list")
        keys_in_order = [i.get("key") for i in items if i.get("type")=="section"]
        if keys_in_order != SECTION_KEYS: raise ValueError("sections order invalid")
        return items
    except Exception:
        # 폴백: 섹션 전부 + 피겨를 5/6에 분산
        base = [{"type":"section","key":k} for k in SECTION_KEYS]
        m = len(figures)
        left = [{"type":"figure","id":f["id"]} for f in figures[:m//2]]
        right= [{"type":"figure","id":f["id"]} for f in figures[m//2:]]
        out=[]
        for it in base:
            out.append(it)
            if it["key"]=="5. 제안 시스템/방법": out.extend(left)
            if it["key"]=="6. 실험 가설/절차": out.extend(right)
        return out

# ── Figure 추출
def get_text_blocks(page: "fitz.Page"):
    blocks = page.get_text("blocks"); out=[]
    for b in blocks:
        x0,y0,x1,y1,text = b[:5]; btype = b[6] if len(b)>6 else 0
        out.append({"rect":fitz.Rect(x0,y0,x1,y1),"text":text or "","type":btype})
    return out
def is_caption_text(text: str) -> bool:
    first = text.strip().splitlines()[0] if text.strip() else ""
    return bool(CAPTION_RE.search(first))
def x_overlap_ratio(a, b):
    left, right = max(a.x0,b.x0), min(a.x1,b.x1)
    if right<=left: return 0.0
    return (right-left)/max(a.width,b.width)
def rect_union(a,b): return fitz.Rect(min(a.x0,b.x0),min(a.y0,b.y0),max(a.x1,b.x1),max(a.y1,b.y1))
def expand_rect(r, margin, clip):
    e = fitz.Rect(r.x0-margin, r.y0-margin, r.x1+margin, r.y1+margin)
    e.x0=max(e.x0,clip.x0); e.y0=max(e.y0,clip.y0); e.x1=min(e.x1,clip.x1); e.y1=min(e.y1,clip.y1); return e
def get_visual_rects(page):
    rects=[]
    for info in page.get_images(full=True):
        xref=info[0]
        for r in page.get_image_rects(xref): rects.append(fitz.Rect(r))
    try:
        for d in page.get_drawings():
            r=d.get("rect")
            if not r: continue
            rr=fitz.Rect(r)
            if rr.width<8 or rr.height<8: continue
            rects.append(rr)
    except Exception: pass
    return rects
def group_for_caption(caption_rect, rects):
    cand=[]
    for r in rects:
        above=(r.y1<=caption_rect.y0) and ((caption_rect.y0-r.y1)<=MAX_VGAP)
        below=(r.y0>=caption_rect.y1) and ((r.y0-caption_rect.y1)<=MAX_VGAP)
        xok = x_overlap_ratio(r, caption_rect) >= MIN_X_OVERLAP_RATIO
        if xok and (above or below): cand.append(r)
    if not cand: return []
    if MERGE_NEAR_IMAGES:
        u=cand[0]
        for r in cand[1:]: u=rect_union(u,r)
        return [u]
    return cand
def render_rect_png_bytes(page, rect, scale=SCALE):
    mat=fitz.Matrix(scale,scale); pix=page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    return pix.tobytes("png")

def extract_and_upload_figures(pdf_path: str) -> list[dict]:
    if fitz is None: return []
    out=[]; out_dir=os.path.join(os.path.dirname(pdf_path), "_figures")
    if SAVE_LOCAL_FIGS: os.makedirs(out_dir, exist_ok=True)
    doc=fitz.open(pdf_path)
    for pno in range(len(doc)):
        page=doc[pno]; page_rect=page.rect
        caps=[b for b in get_text_blocks(page) if b["type"]==0 and is_caption_text(b["text"])]
        if not caps: continue
        vrects=get_visual_rects(page)
        if not vrects: continue
        for cidx, cb in enumerate(caps, start=1):
            cap=cb["rect"]; near=group_for_caption(cap, vrects)
            if not near: continue
            uni=cap
            for rr in near: uni=rect_union(uni, rr)
            crop=expand_rect(uni, MARGIN, page_rect)
            png_bytes=render_rect_png_bytes(page, crop, SCALE)
            fname=f"p{pno+1:02d}_figure{cidx:02d}.png"
            if SAVE_LOCAL_FIGS:
                with open(os.path.join(out_dir, fname),"wb") as f: f.write(png_bytes)
            created=create_file_upload(fname, "image/png"); fid=created.get("id")
            if not fid: continue
            sent=send_file_upload_bytes(fid, png_bytes, fname, "image/png")
            if sent.get("status")!="uploaded": continue
            out.append({
                "id": f"p{pno+1:02d}_f{cidx:02d}",
                "page": pno+1,
                "caption_text": cb["text"].strip(),
                "upload_id": fid,
                "image_block": image_block_from_upload(fid, caption=""),
            })
    doc.close(); return out

# ── 페이지 생성
def pick_first_figure_id(figures: list[dict]) -> str | None:
    if not figures: return None
    return sorted(figures, key=lambda f: (int(f.get("page", 10**9) or 10**9), str(f.get("id",""))))[0].get("id")

def create_notion_page_with_layout(meta, contents, layout, figures_by_id, paper_url, pdf_block, file_hint):
    prop_map = ensure_db_property_map(DATABASE_ID)
    title_prop = prop_map["Title"]
    title_val  = meta.get("title") or os.path.splitext(os.path.basename(file_hint))[0]

    props = {
        title_prop: {"title": [{"text": {"content": s(title_val)}}]},
        prop_map["Tag"]: {"multi_select": [{"name": x.strip()} for x in (meta.get("tag") or [])] if isinstance(meta.get("tag"), (list,tuple)) else [{"name": t.strip()} for t in str(meta.get("tag") or "").split(",") if t.strip()]},
        prop_map["Authors"]: {"rich_text": [{"text": {"content": s(meta.get("authors"))}}]},
        prop_map["Year"]: {"rich_text": [{"text": {"content": s(meta.get("year"))}}]},
        prop_map["Conference/journal"]: {"rich_text": [{"text": {"content": s(meta.get("conference_journal"))}}]},
        prop_map["One sentence"]: {"rich_text": [{"text": {"content": s(meta.get("one_sentence",""))}}]},
    }

    children=[]
    if paper_url: children.append({"object":"block","type":"bookmark","bookmark":{"url":paper_url}})
    if pdf_block: children.append(pdf_block)

    lead_id=None
    if PROMOTE_FIRST_FIGURE and figures_by_id:
        all_fig_list = [{"id": fid, **rec} for fid, rec in figures_by_id.items()]
        lead_id = pick_first_figure_id(all_fig_list)
        if lead_id and figures_by_id.get(lead_id):
            children.append(figures_by_id[lead_id]["image_block"])

    def _heading(text, level=1):
        key={1:"heading_1",2:"heading_2",3:"heading_3"}[max(1,min(level,3))]
        return {"object":"block","type":key, key:{"rich_text":[{"type":"text","text":{"content":text}}]} }
    def _paragraph_rich(parts): return {"object":"block","type":"paragraph","paragraph":{"rich_text":parts}}

    def md_to_rich_text_local(md): return md_to_rich_text(md)

    for item in layout:
        if item.get("type")=="section":
            key=item.get("key"); body=s(contents.get(key,"")).strip()
            if not body: continue
            children.append(_heading(key,1))
            rich=md_to_rich_text_local(body)
            if rich: children.append(_paragraph_rich(rich))
        elif item.get("type")=="figure":
            fid=item.get("id"); rec=figures_by_id.get(fid)
            if rec:
                if PROMOTE_FIRST_FIGURE and lead_id and fid==lead_id: continue
                children.append(rec["image_block"])

    used_keys={i.get("key") for i in layout if i.get("type")=="section"}
    for key in SECTION_KEYS:
        if key not in used_keys and s(contents.get(key,"")).strip():
            children.append(_heading(key,1))
            children.append(_paragraph_rich(md_to_rich_text_local(contents[key])))

    return notion.pages.create(parent={"database_id": DATABASE_ID}, properties=props, children=children)

# ── 파이프라인
def process_pdf(path: str):
    print(f"📄 Processing: {path}")
    text = strip_unpaired_surrogates(read_pdf(path))

    try: meta = call_gpt_meta(text)
    except Exception as e:
        print(f"⚠️ GPT(meta) 오류: {e}"); meta = {"title":"","authors":"","year":"","conference_journal":"","tag":[]}
    if not s(meta.get("year")):
        g=safe_year_guess(os.path.basename(path), text)
        if g: meta["year"]=g

    try: contents = call_gpt_contents_marked(text)
    except Exception as e:
        print(f"⚠️ GPT(contents) 오류: {e}"); contents={k:"" for k in SECTION_KEYS}

    try: meta["one_sentence"] = call_gpt_one_sentence(text)
    except Exception: meta["one_sentence"]=""

    try: paper_url = find_paper_url(meta.get("title",""), meta.get("authors",""), meta.get("year",""))
    except Exception: paper_url=""

    pdf_block=None
    try:
        created = create_file_upload(os.path.basename(path), "application/pdf"); fid=created.get("id")
        sent = send_file_upload_path(fid, path, "application/pdf")
        if sent.get("status")=="uploaded": pdf_block = file_block_from_upload(fid)
    except Exception as e:
        print(f"⚠️ PDF 업로드 실패: {e}")

    try: figures = extract_and_upload_figures(path)
    except Exception as e:
        print(f"⚠️ Figure 추출/업로드 실패: {e}"); figures=[]
    figures_by_id = {f["id"]: f for f in figures}

    try: layout = call_gpt_layout(contents, figures)
    except Exception as e:
        print(f"⚠️ 레이아웃 생성 오류: {e}")
        base=[{"type":"section","key":k} for k in SECTION_KEYS]
        m=len(figures); left=[{"type":"figure","id":f["id"]} for f in figures[:m//2]]
        right=[{"type":"figure","id":f["id"]} for f in figures[m//2:]]
        layout=[]
        for it in base:
            layout.append(it)
            if it["key"]=="5. 제안 시스템/방법": layout.extend(left)
            if it["key"]=="6. 실험 가설/절차": layout.extend(right)

    try:
        res = create_notion_page_with_layout(meta, contents, layout, figures_by_id, paper_url, pdf_block, file_hint=path)
        print(f"✅ Added to Notion: {res.get('url')}\n")
    except APIResponseError as e:
        print(f"❌ Notion API 오류: {e}\n")

def main():
    pdf_dir = os.path.join(os.getcwd(), "paper")
    if not os.path.isdir(pdf_dir):
        print(f"❌ 폴더가 없습니다: {pdf_dir}"); sys.exit(1)
    pdfs = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdfs:
        print("❌ paper 폴더에 PDF가 없습니다."); sys.exit(1)
    for p in pdfs: process_pdf(p)

if __name__ == "__main__":
    main()
