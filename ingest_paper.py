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
USE_GPT_API    = os.getenv("USE_GPT_API", "true").lower() == "true"   # [MODIFIED]
NOTION_VERSION = "2022-06-28"
OPENAI_MAX_TOKENS_META = int(os.getenv("OPENAI_MAX_TOKENS_META", "700"))
OPENAI_MAX_TOKENS_CONTENTS = int(os.getenv("OPENAI_MAX_TOKENS_CONTENTS", "2000"))
OPENAI_MAX_TOKENS_ONESENTENCE = int(os.getenv("OPENAI_MAX_TOKENS_ONESENTENCE", "64"))
OPENAI_MAX_TOKENS_LAYOUT= int(os.getenv("OPENAI_MAX_TOKENS_LAYOUT", "2000"))


_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")



if not (NOTION_TOKEN and DATABASE_ID and OPENAI_API_KEY):
    print("❌ .env에 NOTION_TOKEN / NOTION_DATABASE_ID / OPENAI_API_KEY가 필요합니다.")
    sys.exit(1)

notion = NotionClient(auth=NOTION_TOKEN, notion_version=NOTION_VERSION)
oai    = OpenAI(api_key=OPENAI_API_KEY) if USE_GPT_API else None      # [MODIFIED]

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


def read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read()
        # BOM/널 제거 (있어도 깔끔하게)
        return s.replace("\ufeff", "").replace("\x00", "")
    except FileNotFoundError:
        return ""

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")

def render_prompt(template_str: str, **vars) -> str:
    """
    템플릿 내 {{KEY}} 자리표시자를 vars["KEY"] 값으로 치환.
    예: render_prompt('Hello {{NAME}}', NAME='world') -> 'Hello world'
    """
    def _sub(m):
        key = m.group(1)
        if key in vars:
            return str(vars[key])
        # 치환값 없으면 원문 유지(디버깅 원하면 여기서 에러/로그 처리)
        return m.group(0)
    return _PLACEHOLDER_RE.sub(_sub, template_str)

def load_prompt_from_files(env_key_filename: str, default_filename: str, base_dir: str | None = None) -> str:
    """
    .env에서 파일명 키(env_key_filename)로 파일명을 읽고, base_dir(기본 PROMPT_DIR)과 합쳐 내용을 로드.
    파일 없으면 "" 반환.
    """
    if base_dir is None:
        base_dir = os.getenv("PROMPT_DIR", "./prompts")
    filename = os.getenv(env_key_filename, default_filename)
    path = os.path.join(base_dir, filename)
    return read_text_file(path)



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
        print(db)
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
        text = text[:max_chars]
        # ▶ 백매터 제거
        text = trim_trailing_backmatter(text)
        return text
    except Exception as e:
        return f"[PDF READ ERROR] {e}"

def trim_trailing_backmatter(text: str) -> str:
    """
    1) 'References/Bibliography/Appendix/Acknowledgments' 같은 헤딩을 찾아 그 지점부터 잘라냄.
    2) 못 찾으면 하단부에서 참고문헌스러운 라인 비율을 보고 컷오프.
    3) 과도 트림 방지: 전체의 60% 이전에서 자르지 않음.
    ENV:
      TRIM_BACKMATTER (true/false), BACKMATTER_MIN_POS_RATIO (0~1), BACKMATTER_TAIL_LINES (int)
    """
    if not text:
        return text

    enabled = os.getenv("TRIM_BACKMATTER", "true").lower() == "true"
    if not enabled:
        return text

    import regex as re  # 더 강한 정규식 엔진이 있으면 좋지만, 없으면 re로 바꿔도 OK
    MIN_POS_RATIO = float(os.getenv("BACKMATTER_MIN_POS_RATIO", "0.60"))  # 본문 60% 이후만 컷 허용
    TAIL_LINES = int(os.getenv("BACKMATTER_TAIL_LINES", "400"))           # 아래쪽 스캔 라인 수
    body_len = len(text)
    min_pos = int(body_len * MIN_POS_RATIO)

    # 1) 명시적 헤딩 매치 (멀티라인)
    heading_re = re.compile(
        r"(?m)^\s*(references|bibliography|appendix|appendices|acknowledg?e?ments)\s*$",
        re.IGNORECASE
    )
    m = heading_re.search(text)
    if m and m.start() >= min_pos:
        return text[:m.start()].rstrip()

    # 2) 아래쪽에서 위로 참고문헌 라인 비율 체크
    #    패턴 예: [12] Foo..., 12. Bar..., (2019), 2019., doi:, arXiv:, Proc. of ...
    ref_like_bul = re.compile(
        r"^\s*(\[\d{1,3}\]|\d{1,3}\.|•|-)\s+.+", re.IGNORECASE
    )
    year_like = re.compile(r"\b(19[6-9]\d|20[0-4]\d)\b")
    doi_like  = re.compile(r"\b(doi:|https?://(dx\.)?doi\.org/|arxiv:)\b", re.IGNORECASE)
    proc_like = re.compile(r"\b(proceedings|proc\.|vol\.|no\.|pp\.)\b", re.IGNORECASE)

    lines = text.splitlines()
    n = len(lines)
    start_line = max(0, n - TAIL_LINES)
    tail = lines[start_line:]

    def is_ref_line(line: str) -> bool:
        l = line.strip()
        if not l:
            return False
        hits = 0
        if ref_like_bul.match(l): hits += 1
        if year_like.search(l):   hits += 1
        if doi_like.search(l):    hits += 1
        if proc_like.search(l):   hits += 1
        # 기준: 위 특징 중 2개 이상
        return hits >= 2

    # 아래쪽에서 위로 가면서 "연속적으로 참고문헌 같은" 구간의 시작 지점 찾기
    ref_run = 0
    run_needed = 6  # 최소 6줄 정도 연속으로 참고문헌 느낌이면 컷
    cutoff_idx_in_tail = None

    for i in range(len(tail) - 1, -1, -1):
        if is_ref_line(tail[i]):
            ref_run += 1
        else:
            if ref_run >= run_needed:
                cutoff_idx_in_tail = i + 1
                break
            ref_run = 0

    # 끝까지 왔을 때도 긴 러닝이 유지되면 tail 시작에서 컷
    if cutoff_idx_in_tail is None and ref_run >= run_needed:
        cutoff_idx_in_tail = 0

    if cutoff_idx_in_tail is not None:
        abs_cut = start_line + cutoff_idx_in_tail
        # 과도 트림 방지
        cut_pos = sum(len(l) + 1 for l in lines[:abs_cut])
        if cut_pos >= min_pos:
            return "\n".join(lines[:abs_cut]).rstrip()

    return text

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

def call_gpt_meta(text_snippet: str) -> dict:
    if not USE_GPT_API:
        print("[META] USE_GPT_API=False → 테스트값 반환")
        return {
            "title": "Test Paper Title",
            "authors": "Doe, J.; Smith, A.",
            "year": "2023",
            "conference_journal": "CHI Conference",
            "tag": ["HCI", "haptics", "test"],
        }

    # 0) 입력 전처리/로그
    text_snippet = strip_unpaired_surrogates(text_snippet)
    #print("[META] snippet_head:", (text_snippet[:300] + " …") if len(text_snippet) > 300 else text_snippet)

    # 1) 프롬프트 파일 경로/내용 로드
    base_dir  = os.getenv("PROMPT_DIR", "./prompts")
    sys_path  = os.path.join(base_dir, os.getenv("PROMPT_META_SYSTEM", "meta.system.txt"))
    user_path = os.path.join(base_dir, os.getenv("PROMPT_META_USER",   "meta.user.txt"))
    #print("[META] system_path:", sys_path)
    #print("[META] user_path  :", user_path)

    system_msg = read_text_file(sys_path)
    user_tpl   = read_text_file(user_path)

    if not system_msg:
        system_msg = "너는 논문 메타데이터 추출 어시스턴트다. JSON만 반환하라."
        print("[META][WARN] system prompt 파일 비어있음 → 기본 문구 사용")
    if not user_tpl:
        user_tpl = (
            "다음 스니펫으로 title, authors, year, conference_journal, tag만 포함한 JSON을 만들어라.\n"
            "\"\"\"{{TEXT_SNIPPET}}\"\"\""
        )
        print("[META][WARN] user prompt 파일 비어있음 → 기본 템플릿 사용")

    user_msg = render_prompt(user_tpl, TEXT_SNIPPET=text_snippet)
    #print("[META] system_msg_head:", (system_msg[:200] + " …") if len(system_msg) > 200 else system_msg)
    #print("[META] user_msg_head  :", (user_msg[:200] + " …") if len(user_msg) > 200 else user_msg)

    # 2) 호출
    resp = oai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
        max_tokens=OPENAI_MAX_TOKENS_META,
    )
    raw = resp.choices[0].message.content or "{}"
    #print("[META] raw_resp:", raw)

    # 3) 파싱(래핑/대소문자/동의어 보강)
    try:
        data = json.loads(raw)
    except Exception as e:
        print("[META][ERROR] JSON 파싱 실패:", e)
        return {"title":"", "authors":"", "year":"", "conference_journal":"", "tag":[], "_raw": raw}

    # 응답이 {"meta": {...}} 이거나 바로 {...} 일 수 있음
    payload = data.get("meta") if isinstance(data, dict) else None
    if not isinstance(payload, dict):
        payload = data if isinstance(data, dict) else {}

    # 키 정규화/동의어 맵핑
    def _norm(s): return (s or "").strip().lower().replace(" ", "").replace("-", "_")
    norm_map = { _norm(k): k for k in ["title","authors","year","conference_journal","tag"] }

    # 동의어 → 표준키
    syn = {
        "conference": "conference_journal",
        "venue": "conference_journal",
        "journal": "conference_journal",
        "conf": "conference_journal",
    }

    out = {"title":"", "authors":"", "year":"", "conference_journal":"", "tag":[]}

    for k, v in payload.items():
        nk = _norm(k)
        if nk in norm_map:
            key = norm_map[nk]
        elif nk in syn:
            key = syn[nk]
        else:
            continue  # 알 수 없는 키는 무시

        # 값 후처리
        if key == "tag":
            if isinstance(v, (list, tuple, set)):
                out["tag"] = [str(x).strip() for x in v if str(x).strip()]
            elif isinstance(v, str):
                out["tag"] = [x.strip() for x in v.split(",") if x.strip()]
            else:
                out["tag"] = []
        elif key == "authors":
            if isinstance(v, (list, tuple)):
                out["authors"] = "; ".join([str(x).strip() for x in v if str(x).strip()])
            else:
                out["authors"] = str(v).strip()
        else:
            out[key] = str(v).strip()

    # 4) 필드별 상태 로그
    # print("[META] parsed =>",
    #       "title=", bool(out["title"]), 
    #       "authors=", bool(out["authors"]),
    #       "year=", out["year"],
    #       "conf_journal=", bool(out["conference_journal"]),
    #       "tag_len=", len(out["tag"]))

    # 5) 비어있는 필드 보조 추출(약식)
    #  - PDF metadata에서도 종종 제목/저자 나옴
    #  - 파일명/본문에서 연도 추정은 이미 elsewhere에서 수행
    if (not out["title"] or not out["authors"]) and 'PdfReader' in globals():
        # (선택) 필요 시 여기에 reader.metadata 추출 로직 추가 가능
        pass

    # 6) 최종 반환
    return out





def call_gpt_contents_marked(text_snippet: str) -> dict:
    if not USE_GPT_API:
        return {k: f"{k} : 테스트용 임의 내용입니다." for k in SECTION_KEYS}

    text_snippet = strip_unpaired_surrogates(text_snippet)

    base_dir  = os.getenv("PROMPT_DIR", "./prompts")
    sys_path  = os.path.join(base_dir, os.getenv("PROMPT_CONTENTS_SYSTEM", "contents.system.txt"))
    user_path = os.path.join(base_dir, os.getenv("PROMPT_CONTENTS_USER",   "contents.user.txt"))

    #print(sys_path)
    

    system_msg = read_text_file(sys_path)
    user_tpl   = read_text_file(user_path)

    # 🔁 여기서 실제 논문 본문을 {{TEXT_SNIPPET}}에 꽂아 넣음
    user_msg = render_prompt(user_tpl, TEXT_SNIPPET=text_snippet)
    #print("[CONTENTS user_msg]\\n", user_msg[:600])

    # (옵션) 치환이 안 된 자리표시자 남아있으면 경고/예외로 잡아내도 좋음
    # if "{{TEXT_SNIPPET}}" in user_msg: raise ValueError("TEXT_SNIPPET 치환 실패: 템플릿/키 확인")

    # 폴백
    if not system_msg:
        system_msg = "너는 논문 요약에 특화된 한국어 연구 비서다. 반드시 유효한 JSON만 반환해라."
    if not user_msg:
        user_msg = '다음 스니펫으로 "contents" 딕셔너리를 만들어라:\n"""' + text_snippet + '"""'

    resp = oai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
        max_tokens=OPENAI_MAX_TOKENS_CONTENTS,
    )
    raw = resp.choices[0].message.content
    #print("[CONTENTS raw]\\n", raw)
    data = json.loads(raw)
    
    contents = data.get("contents", data)
    #print("[CONTENTS keys]", list(contents.keys()))
    for k in SECTION_KEYS:
        contents.setdefault(k, "")
    return contents



def call_gpt_one_sentence(text_snippet: str) -> str:
    if not USE_GPT_API:
        return "테스트용 한줄 요약입니다."

    text_snippet = strip_unpaired_surrogates(text_snippet)

    base_dir  = os.getenv("PROMPT_DIR", "./prompts")
    sys_path  = os.path.join(base_dir, os.getenv("PROMPT_ONELINE_SYSTEM", "oneline.system.txt"))
    user_path = os.path.join(base_dir, os.getenv("PROMPT_ONELINE_USER",   "oneline.user.txt"))

    system_msg = read_text_file(sys_path) or "한 문장(30자 이내)만 반환하라."
    user_tpl   = read_text_file(user_path) or '다음 텍스트를 30자 이내 한 문장으로 요약:\n"""{{TEXT_SNIPPET}}"""'

    user_msg = render_prompt(user_tpl, TEXT_SNIPPET=text_snippet)
    resp = oai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=OPENAI_MAX_TOKENS_ONESENTENCE,
    )
    return (resp.choices[0].message.content or "").strip().replace("\n", " ")


def call_gpt_layout(contents: dict, figures: list[dict]) -> list[dict]:
    """
    목표:
    - 섹션 순서(1→6) 고정
    - 모든 figure를 정확히 1회씩 '관련 섹션 바로 뒤'에 배치 (GPT가 빼먹어도 코드가 강제 보정)
    - 캡션/섹션 요약은 기본적으로 '전체' 사용 (ENV로 제한 가능)
    - 풍부한 디버그 로그 출력
    """

    # ====== 0) ENV로 길이 제어 (기본: 전체 사용) ======
    try:
        MAX_SECTION_CHARS = int(os.getenv("PROMPT_SECTION_CHARS", "0"))  # 0이면 전체
        MAX_CAPTION_CHARS = int(os.getenv("PROMPT_CAPTION_CHARS", "0"))  # 0이면 전체
    except Exception:
        MAX_SECTION_CHARS = 0
        MAX_CAPTION_CHARS = 0

    def _brief_full(s: str, n: int = 0) -> str:
        s = strip_unpaired_surrogates(s or "")
        return s if n <= 0 else s[:n]

    # ====== 1) 입력 정리 (섹션/피겨 본문 그대로 투입) ======
    contents_brief = {k: _brief_full(contents.get(k, ""), MAX_SECTION_CHARS) for k in SECTION_KEYS}
    fig_brief = [{
        "id": f.get("id"),
        "page": int(f.get("page", 0) or 0),
        # ▶ 캡션 전체 (자르지 않음; 필요시 ENV로 제한)
        "caption": _brief_full(f.get("caption_text", ""), MAX_CAPTION_CHARS)
    } for f in figures]

    # ====== 2) 입력 로그 ======
    #print("\n[LAYOUT] === SECTION BRIEF (요약/전체) ===")
    for k in SECTION_KEYS:
        txt = contents_brief.get(k, "")
        #print(f"\n--- {k} ---\n{txt[:600]}{'...' if len(txt)>600 else ''}")

    #print("\n[LAYOUT] === FIGURE BRIEF (캡션) ===")
    for fb in fig_brief:
        cap = fb.get("caption","")
        #print(f"\n- {fb['id']}: {cap[:600]}{'...' if len(cap)>600 else ''}")

    # ====== 3) GPT 호출 ======
    base_dir  = os.getenv("PROMPT_DIR", "./prompts")
    sys_path  = os.path.join(base_dir, os.getenv("PROMPT_LAYOUT_SYSTEM", "layout.system.txt"))
    user_path = os.path.join(base_dir, os.getenv("PROMPT_LAYOUT_USER",   "layout.user.txt"))

    system_msg = read_text_file(sys_path) or "너는 Notion 페이지 레이아웃을 설계하는 시스템이다. 반드시 JSON 객체만 반환하라."
    user_tpl   = read_text_file(user_path) or (
        # SECTION_KEYS도 템플릿에 넣어주기
        "필수 규칙:\n1) 6개 섹션을 이 순서로 모두 포함: {{SECTION_KEYS}}\n"
        "2) 모든 피겨 id를 정확히 한 번씩 포함\n\n"
        "섹션 요약:\n{{CONTENTS_BRIEF_JSON}}\n\n피겨 목록:\n{{FIG_BRIEF_JSON}}"
    )

    user_msg = render_prompt(
        user_tpl,
        SECTION_KEYS=json.dumps(SECTION_KEYS, ensure_ascii=False),
        CONTENTS_BRIEF_JSON=json.dumps(contents_brief, ensure_ascii=False, indent=2),
        FIG_BRIEF_JSON=json.dumps(fig_brief, ensure_ascii=False, indent=2),
    )

    items_from_gpt = []
    if USE_GPT_API:
        try:
            resp = oai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
                temperature=0.2,
                response_format={"type":"json_object"},
                max_tokens=OPENAI_MAX_TOKENS_LAYOUT,
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
            items_from_gpt = data.get("items", []) if isinstance(data, dict) else []
        except Exception as e:
            print(f"[LAYOUT][ERROR] GPT 호출/파싱 실패: {e}")
    else:
        print("[LAYOUT] USE_GPT_API=False → 임시 빈 레이아웃 사용")

    # ====== 4) 정합성/클린업 (GPT 결과를 최대한 살리되, 누락 figure는 채움) ======
    valid_ids = {f["id"] for f in figures}
    # figure → 섹션 버킷
    buckets: dict[str, list[str]] = {k: [] for k in SECTION_KEYS}
    used_figs: set[str] = set()

    # (A) GPT가 준 items를 훑어서, 섹션 바로 뒤에 온 figure만 해당 섹션 버킷에 담기
    last_section = None
    for it in items_from_gpt:
        t = it.get("type")
        if t == "section" and it.get("key") in SECTION_KEYS:
            last_section = it["key"]
        elif t == "figure":
            fid = it.get("id")
            if fid in valid_ids and fid not in used_figs and last_section in buckets:
                buckets[last_section].append(fid)
                used_figs.add(fid)
            else:
                # 무효/중복/섹션컨텍스트 없음 → 일단 패스(아래에서 보정)
                pass

    # (B) 누락 figure들 찾아서 휴리스틱으로 섹션 배정 (무조건 전부 포함되도록)
    def _heuristic_section(cap: str) -> str:
        c = (cap or "").lower()
        # 실험/측정/결과/평가 → 섹션 6
        if any(w in c for w in ["experiment", "user study", "measurement", "measured",
                                "result", "evaluation", "participants", "task", "procedure", "accuracy", "comparison"]):
            return "6. 실험 가설/절차"
        # 방법/시스템/아키텍처 → 섹션 5
        if any(w in c for w in ["method", "system", "architecture", "pipeline",
                                "approach", "algorithm", "implementation", "design", "controller", "circuit"]):
            return "5. 제안 시스템/방법"
        # 선행/배경 → 섹션 2
        if any(w in c for w in ["related", "prior", "previous", "background", "state-of-the-art"]):
            return "2. 선행연구"
        # 개선점/중요성 키워드
        if any(w in c for w in ["improvement", "advantage", "novelty", "contribution"]):
            return "3. 이번 연구의 개선점"
        if any(w in c for w in ["motivation", "importance", "significance"]):
            return "4. 문제의 중요성"
        # 폴백: 5 (방법)
        return "5. 제안 시스템/방법"

    missing_ids = [fid for fid in valid_ids if fid not in used_figs]
    if missing_ids:
        print(f"[LAYOUT][WARN] GPT가 누락한 figure 수: {len(missing_ids)} → 휴리스틱으로 채움")
    # 캡션 맵
    cap_by_id = {fb["id"]: fb.get("caption","") for fb in fig_brief}
    for fid in missing_ids:
        sec = _heuristic_section(cap_by_id.get(fid, ""))
        buckets[sec].append(fid)
        used_figs.add(fid)
        # 로그: 어떤 섹션으로 보정됐는지
        cap = cap_by_id.get(fid, "")
        print(f"[LAYOUT][FILL] figure={fid} → section='{sec}' by heuristic")
        cap_snippet = cap[:200].replace("\n", " ")
        print(f"  • caption: {cap_snippet}{'...' if len(cap) > 200 else ''}")


    # ====== 5) 최종 items: 1→6 섹션을 고정 순서로 깔고, 각 섹션 뒤에 버킷 figure를 연달아 삽입 ======
    final_items: list[dict] = []
    for sec in SECTION_KEYS:
        final_items.append({"type":"section","key":sec})
        for fid in buckets[sec]:
            final_items.append({"type":"figure","id":fid})

    # ====== 6) 최종 시퀀스 로그 + 커버리지 검증 ======
    seq = ["{sec:"+it["key"]+"}" if it["type"]=="section" else "{fig:"+it["id"]+"}" for it in final_items]
    print("\n[LAYOUT] === FINAL ITEMS ORDER ===")
    print(" ".join(seq))

    placed_figs = [it["id"] for it in final_items if it["type"]=="figure"]
    if set(placed_figs) != valid_ids or len(placed_figs) != len(valid_ids):
        print("[LAYOUT][ERROR] figure 커버리지가 100%가 아님! (논리 점검 필요)")
        print("  placed:", placed_figs)
        print("  valid :", sorted(valid_ids))

    return final_items



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
    print(text)
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

    for k in SECTION_KEYS:
        print("LEN", k, len(s(contents.get(k, ""))))
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
