# ingest_papers.py
# ============================================================
# PDF -> GPT(메타/본문/한줄요약) -> Notion DB 기록
# + Notion file_uploads REST로 PDF/이미지 직접 업로드
# + 논문 URL Bookmark + (섞기 레이아웃) Sections & Figures + Contents
# ============================================================

# 0) Imports
import os
import re
import glob
import json
import sys
import io
import mimetypes
from datetime import datetime
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
    
# 상단에 추가할지 여부 (전역)
PROMOTE_FIRST_FIGURE = True

def pick_first_figure_id(figures: list[dict]) -> str | None:
    """
    가장 이른 figure를 선택 (page 오름차순, 동점이면 id 사전순)
    """
    if not figures:
        return None
    return sorted(figures, key=lambda f: (int(f.get("page", 10**9) or 10**9), str(f.get("id",""))))[0].get("id")


# 1) Env / Clients
load_dotenv()
NOTION_TOKEN   = os.getenv("NOTION_TOKEN")
DATABASE_ID    = os.getenv("NOTION_DATABASE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "chatgpt-4o-latest")
NOTION_VERSION = "2022-06-28"

OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "700"))
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def strip_unpaired_surrogates(s: str) -> str:
    """
    문자열에서 고아 서러게이트를 제거한다.
    - 정상적인 UTF-16 쌍은 보존됨 (encode/decode 트릭)
    """
    if not isinstance(s, str):
        return s
    # 1) 빠른 경로: 명시적 제거(고아 포함 영역 전부 삭제)
    s = _SURROGATE_RE.sub("", s)
    # 2) 보수적 복원: 유효한 쌍은 유지, 고아는 drop
    #   (surrogatepass로 16비트 단위 그대로 인코드 → 유효하지 않은 조합만 decode에서 소거)
    try:
        s = s.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")
    except Exception:
        # 환경에 따라 위 단계가 불편하면, utf-8 ignore fallback
        s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
    # 널문자 등 통신에 거슬리는 제어문자도 정리
    s = s.replace("\x00", "")
    return s

def sanitize_text(obj):
    """
    dict/list/str 재귀 순회하면서 모든 문자열에 strip_unpaired_surrogates 적용
    """
    if isinstance(obj, str):
        return strip_unpaired_surrogates(obj)
    if isinstance(obj, list):
        return [sanitize_text(x) for x in obj]
    if isinstance(obj, dict):
        return {k: sanitize_text(v) for k, v in obj.items()}
    return obj

if not (NOTION_TOKEN and DATABASE_ID and OPENAI_API_KEY):
    print("❌ .env에 NOTION_TOKEN / NOTION_DATABASE_ID / OPENAI_API_KEY가 필요합니다.")
    sys.exit(1)

notion = NotionClient(auth=NOTION_TOKEN)
oai    = OpenAI(api_key=OPENAI_API_KEY)


# 2) Parameters (Figure 탐지/크롭)
SCALE      = 2.0
MARGIN     = 6.0
MAX_VGAP   = 120.0
MIN_X_OVERLAP_RATIO = 0.20
MERGE_NEAR_IMAGES   = True
SAVE_LOCAL_FIGS     = False  # True면 _figures/ 저장도 수행

CAPTION_RE = re.compile(
    r"^\s*(?:"
    r"(?:Figure|Fig\.?)\s*[0-9IVX]+|"
    r"(?:그림|도)\s*\d+"
    r")\b",
    re.IGNORECASE
)

SECTION_KEYS = [
    "1. 기존문제",
    "2. 선행연구",
    "3. 이번 연구의 개선점",
    "4. 문제의 중요성",
    "5. 제안 시스템/방법",
    "6. 실험 가설/절차",
]


# 3) Utilities
def s(val) -> str:
    return "" if val is None else str(val)

def guess_mime(path: str, default="application/octet-stream"):
    mime, _ = mimetypes.guess_type(path)
    return mime or default

def ensure_title_prop_name(dbid: str) -> str:
    db = notion.databases.retrieve(dbid)
    for k, v in db["properties"].items():
        if v["type"] == "title":
            return k
    raise RuntimeError("Notion DB에 title 속성이 없습니다.")

def read_pdf(path: str, max_chars: int = 40000) -> str:
    try:
        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            chunk = page.extract_text() or ""
            if chunk:
                texts.append(chunk)
            if sum(len(t) for t in texts) >= max_chars:
                break
        text = "\n\n".join(texts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) > max_chars:
            text = text[:max_chars]
        return text
    except Exception as e:
        return f"[PDF READ ERROR] {e}"

def safe_year_guess(filename: str, text: str) -> str:
    cand = re.findall(r"\b(19[7-9]\d|20[0-4]\d|2025|2026)\b", filename + " " + text)
    return cand[0] if cand else ""

def to_tag_list(val) -> list[dict]:
    if val is None:
        return []
    if isinstance(val, str):
        parts = [x.strip() for x in val.split(",") if x.strip()]
    elif isinstance(val, (list, tuple, set)):
        parts = [s(x).strip() for x in val if s(x).strip()]
    else:
        parts = [s(val).strip()]
    return [{"name": x} for x in parts]

def _heading(text: str, level: int = 2) -> dict:
    key = {1: "heading_1", 2: "heading_2", 3: "heading_3"}[max(1, min(level, 3))]
    return {"object": "block", "type": key, key: {"rich_text": [{"type": "text", "text": {"content": text}}]}}

def _paragraph_rich(parts: list[dict]) -> dict:
    return {"object":"block","type":"paragraph","paragraph":{"rich_text":parts}}

def _bookmark(url: str) -> dict:
    return {"object": "block", "type": "bookmark", "bookmark": {"url": url}}

def _numbered_rich(parts: list[dict]) -> dict:
    return {"object":"block","type":"numbered_list_item","numbered_list_item":{"rich_text":parts}}

def _notion_headers(extra: dict | None = None) -> dict:
    base = {"Authorization": f"Bearer {NOTION_TOKEN}", "Notion-Version": NOTION_VERSION}
    if extra: base.update(extra)
    return base


# 4) Notion file_uploads
def create_file_upload(filename: str, content_type: str) -> dict:
    url = "https://api.notion.com/v1/file_uploads"
    payload = {"filename": filename, "content_type": content_type}
    r = requests.post(url, headers=_notion_headers({"Content-Type":"application/json"}), data=json.dumps(payload), timeout=30)
    r.raise_for_status()
    return r.json()

def send_file_upload_bytes(file_upload_id: str, data: bytes, filename: str, content_type: str) -> dict:
    url = f"https://api.notion.com/v1/file_uploads/{file_upload_id}/send"
    files = {"file": (filename, io.BytesIO(data), content_type)}
    r = requests.post(url, headers=_notion_headers(), files=files, timeout=120)
    r.raise_for_status()
    return r.json()

def send_file_upload_path(file_upload_id: str, file_path: str, content_type: str) -> dict:
    url = f"https://api.notion.com/v1/file_uploads/{file_upload_id}/send"
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, content_type)}
        r = requests.post(url, headers=_notion_headers(), files=files, timeout=120)
    r.raise_for_status()
    return r.json()

def file_block_from_upload(file_upload_id: str) -> dict:
    return {"object":"block","type":"file","file":{"type":"file_upload","file_upload":{"id":file_upload_id}}}

def image_block_from_upload(file_upload_id: str, caption: str = "") -> dict:
    block = {"object":"block","type":"image","image":{"type":"file_upload","file_upload":{"id":file_upload_id}}}
    if caption:
        block["image"]["caption"] = [{"type":"text","text":{"content":caption}}]
    return block


# 5) Paper URL finder
def find_paper_url(title: str, authors: str = "", year: str = "") -> str:
    if not title: return ""
    title_q = quote_plus(title.strip())
    # Crossref
    try:
        url = f"https://api.crossref.org/works?query.title={title_q}&rows=3"
        r = requests.get(url, timeout=10)
        if r.ok:
            items = r.json().get("message", {}).get("items", [])
            title_norm = re.sub(r"\s+"," ",title).strip().lower()
            best = None
            for it in items:
                it_title = " ".join(it.get("title", [])).strip()
                it_norm = re.sub(r"\s+"," ",it_title).strip().lower()
                score = 2 if it_norm == title_norm else (1 if title_norm in it_norm or it_norm in title_norm else 0)
                if year and str(it.get("issued", {}).get("date-parts", [[None]])[0][0]) == year: score += 1
                if not best or score > best[0]: best = (score, it)
            if best and best[0] >= 1:
                it = best[1]
                doi = it.get("DOI")
                if doi: return f"https://doi.org/{doi}"
                link = it.get("URL")
                if link: return link
    except Exception: pass
    # arXiv
    try:
        r = requests.get(f"http://export.arxiv.org/api/query?search_query=ti:{title_q}", timeout=10)
        if r.ok and "<entry>" in r.text:
            m = re.search(r"<id>(https?://arxiv\.org/abs/[^\<]+)</id>", r.text)
            if m: return m.group(1).strip()
    except Exception: pass
    # Semantic Scholar
    try:
        s2 = f"https://api.semanticscholar.org/graph/v1/paper/search?query={title_q}&limit=1&fields=url,externalIds,title,year"
        r = requests.get(s2, timeout=10)
        if r.ok and r.json().get("data"):
            d0 = r.json()["data"][0]
            if d0.get("url"): return d0["url"]
            ext = d0.get("externalIds", {})
            if "DOI" in ext: return f"https://doi.org/{ext['DOI']}"
    except Exception: pass
    return ""


# 6) Minimal Markdown(굵게/밑줄/이탤릭) → Notion rich_text 변환
MD_PATTERNS = [
    (re.compile(r"\*\*(.+?)\*\*"), {"bold": True}),
    (re.compile(r"__(.+?)__"),     {"underline": True}),
    (re.compile(r"\*(.+?)\*"),     {"italic": True}),
]

def md_to_rich_text(md: str) -> list[dict]:
    """
    **굵게**, __밑줄__, *이탤릭* 만 지원하는 아주 작은 변환기
    나머지는 일반 텍스트로 처리
    """
    spans = [(0, md, {})]
    for pat, anno in MD_PATTERNS:
        new_spans = []
        for start_idx, text, attrs in spans:
            pos = 0
            for m in pat.finditer(text):
                if m.start() > pos:
                    new_spans.append((0, text[pos:m.start()], attrs))
                new_spans.append((0, m.group(1), {**attrs, **anno}))
                pos = m.end()
            if pos < len(text):
                new_spans.append((0, text[pos:], attrs))
        spans = new_spans

    rich = []
    for _, piece, attrs in spans:
        if not piece:
            continue
        rich.append({
            "type": "text",
            "text": {"content": piece},
            "annotations": {
                "bold": attrs.get("bold", False),
                "italic": attrs.get("italic", False),
                "underline": attrs.get("underline", False),
                "code": False, "strikethrough": False, "color": "default"
            }
        })
    return rich


# 7) GPT Calls
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
1) 스니펫의 "CCS Concepts" 또는 유사 섹션 아래에 있는 키워드/주제 표현을 1~6개 추출한다.
   - 형식의 차이는 무시하고 핵심 명사구로만 담는다.
   - 없으면 논문 주제에 기반해 CCS 스타일의 핵심 키워드를 1~6개 추정해라.
2) 이어서 논문을 빠르게 분류하는 데 유용한 Tag 를 추가해라.
   - 중복 금지
3) 결과적으로 tag는 ["<CCS 또는 추정>", ..., "<분류태그1>", "<분류태그2>", "<분류태그3>"] 형태의 **평탄 배열**이어야 한다.
4) 모든 값은 텍스트만 포함. 설명/괄호/기호는 빼고 핵심어만 넣어라.

스니펫:
\"\"\"{text_snippet}\"\"\""""
    resp = oai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
        temperature=0.2, response_format={"type":"json_object"}, max_tokens=OPENAI_MAX_TOKENS,
    )
    raw = resp.choices[0].message.content
    try: return json.loads(raw)
    except Exception: return {"title":"","authors":"","year":"","conference_journal":"","tag":[],"_raw":raw}

def call_gpt_contents_marked(text_snippet: str) -> dict:
    text_snippet = strip_unpaired_surrogates(text_snippet)
    """
    6개 섹션을 '정확한 키 이름'으로 반환 + 본문은 '간단 마크다운' 포함 허용.
    또한 각 섹션 첫 문장은 '제목 : ' 형태로 시작하도록 지시.
    """
    system_msg = "너는 논문 요약에 특화된 한국어 연구 비서다. 반드시 유효한 JSON만 반환해라."
    user_msg = f"""
다음 논문 스니펫을 바탕으로 'contents' 딕셔너리를 만들어라.
키는 반드시 아래 6개를 정확히 사용한다(철자/구두점 포함, 순서 유지):
- "1. 기존문제"
- "2. 선행연구"
- "3. 이번 연구의 개선점"
- "4. 문제의 중요성"
- "5. 제안 시스템/방법"
- "6. 실험 가설/절차"

각 값은 한국어 단락 문자열이며, 다음 규칙을 지킨다:
- 첫 문장은 '키 이름 : ' 으로 시작한다. 예) "1. 기존문제 : ..."
- 읽기 좋게 **굵게**, __밑줄__, *이탤릭* 같은 마크다운을 적절히 사용한다.
- 각 섹션 길이는 네가 가진 정보로 충분히 요약하되, 핵심 개념에는 굵게나 밑줄을 사용한다.

스니펫:
\"\"\"{text_snippet}\"\"\""""

    resp = oai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
        temperature=0.2, response_format={"type":"json_object"}, max_tokens=2000,
    )
    raw = resp.choices[0].message.content
    data = json.loads(raw)
    # contents가 없는 경우 보정
    if "contents" in data:
        contents = data["contents"]
    else:
        contents = data
    # 누락 키 보강
    for k in SECTION_KEYS:
        contents.setdefault(k, "")
    return contents

def call_gpt_layout(contents: dict, figures: list[dict]) -> list[dict]:
    """
    섹션(1~6)을 반드시 그 순서대로 모두 포함하고,
    각 섹션 뒤에 관련 피겨들을 배치하는 레이아웃을 LLM으로 생성.
    반환 예:
      [{"type":"section","key":"1. 기존문제"},
       {"type":"figure","id":"p01_f02"},
       {"type":"section","key":"2. 선행연구"},
       ...]
    """
    # 1) brief 생성 (길이 제한 + sanitize)
    def _brief(s: str, n: int = 600) -> str:
        return strip_unpaired_surrogates((s or "")[:n])

    # 섹션 요약(너무 길면 잘리도록)
    contents_brief = {k: _brief(contents.get(k, ""), 900) for k in SECTION_KEYS}

    # 피겨 요약(캡션 길이 제한)
    fig_brief = []
    for f in figures:
        fig_brief.append({
            "id": f.get("id"),
            "page": int(f.get("page", 0) or 0),
            "caption": _brief(f.get("caption_text", ""), 400),
        })

    # 2) 프롬프트
    system_msg = "너는 논문 편집 레이아웃 어시스턴트다. 반드시 유효한 JSON만 반환하라."
    user_msg = f"""
아래 '섹션 요약'과 '피겨 목록'을 바탕으로, 독자가 읽기 자연스러운 레이아웃(JSON 배열)을 만들어라.

필수 규칙:
1) 6개 섹션을 이 순서로 모두 포함(누락/순서 변경 금지): {SECTION_KEYS}
2) 피겨는 **관련성이 가장 높은 섹션 바로 뒤에** 배치한다. 
   - 섹션들 사이사이에 자연스럽게 분산 배치하고, 
   - **모든 피겨를 마지막에 몰아서 배치해서는 안 된다.**
3) 각 섹션 항목은 정확히 다음 형태다:
   {{"type":"section","key":"<섹션키>"}}
4) 피겨 항목은 다음 형태다:
   {{"type":"figure","id":"<피겨 id>"}}
5) 피겨는 **관련성이 가장 높은 섹션 바로 뒤**에 배치한다. (여러 개면 그 섹션 뒤에 연달아 배치)
   - 예: 방법/아키텍처 도식 → "5. 제안 시스템/방법" 뒤
   - 실험 결과 그래프 → "6. 실험 가설/절차" 뒤 (또는 결과/평가 내용 뒤)
6) 같은 피겨 id는 한 번만 사용한다.
7) 오직 JSON 배열만 출력한다. 설명/주석 금지.

섹션 요약(내용 일부):
{json.dumps(contents_brief, ensure_ascii=False, indent=2)}

피겨 목록(페이지/캡션 일부):
{json.dumps(fig_brief, ensure_ascii=False, indent=2)}
"""

    # 3) 호출
    resp = oai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role":"system", "content":system_msg},
            {"role":"user",   "content":user_msg},
        ],
        temperature=0.2,
        response_format={"type":"json_object"},
        max_tokens=1200,
    )
    raw = resp.choices[0].message.content

    # 4) 파싱 & 보정
    try:
        layout = json.loads(raw)
        if isinstance(layout, dict) and "items" in layout:
            layout = layout["items"]
        if not isinstance(layout, list):
            raise ValueError("layout is not a list")

        # 최소 검증: 섹션이 6개 모두 순서대로 들어있는지 체크
        keys_in_order = [item.get("key") for item in layout if item.get("type") == "section"]
        if keys_in_order != SECTION_KEYS:
            # 강제 보정: 섹션을 순서대로 재배열하고, figure들은 임시로 뒤에 붙임
            raise ValueError("sections order invalid")
        return layout

    except Exception:
        # 폴백: 섹션 전부(순서대로) + 모든 피겨를 맨 끝에
        fallback = [{"type":"section","key":k} for k in SECTION_KEYS]
        fallback += [{"type":"figure","id":f["id"]} for f in figures]
        return fallback


# 8) Figure detection (vector+raster) & rendering
def is_caption_text(text: str) -> bool:
    first = text.strip().splitlines()[0] if text.strip() else ""
    return bool(CAPTION_RE.search(first))

def get_text_blocks(page: "fitz.Page"):
    blocks = page.get_text("blocks")
    out = []
    for b in blocks:
        x0, y0, x1, y1, text = b[:5]
        btype = b[6] if len(b) > 6 else 0
        out.append({"rect": fitz.Rect(x0,y0,x1,y1), "text": text or "", "type": btype})
    return out

def x_overlap_ratio(a: "fitz.Rect", b: "fitz.Rect") -> float:
    left, right = max(a.x0,b.x0), min(a.x1,b.x1)
    if right <= left: return 0.0
    return (right-left) / max(a.width, b.width)

def rect_union(a: "fitz.Rect", b: "fitz.Rect") -> "fitz.Rect":
    return fitz.Rect(min(a.x0,b.x0), min(a.y0,b.y0), max(a.x1,b.x1), max(a.y1,b.y1))

def expand_rect(r: "fitz.Rect", margin: float, clip: "fitz.Rect") -> "fitz.Rect":
    e = fitz.Rect(r.x0-margin, r.y0-margin, r.x1+margin, r.y1+margin)
    e.x0 = max(e.x0, clip.x0); e.y0 = max(e.y0, clip.y0)
    e.x1 = min(e.x1, clip.x1); e.y1 = min(e.y1, clip.y1)
    return e

def get_visual_rects(page: "fitz.Page"):
    rects = []
    # raster
    for info in page.get_images(full=True):
        xref = info[0]
        for r in page.get_image_rects(xref):
            rects.append(fitz.Rect(r))
    # vector
    try:
        for d in page.get_drawings():
            r = d.get("rect")
            if not r: continue
            rr = fitz.Rect(r)
            if rr.width < 8 or rr.height < 8:
                continue
            rects.append(rr)
    except Exception:
        pass
    return rects

def group_for_caption(caption_rect: "fitz.Rect", rects: list["fitz.Rect"]) -> list["fitz.Rect"]:
    cand = []
    for r in rects:
        above = (r.y1 <= caption_rect.y0) and ((caption_rect.y0 - r.y1) <= MAX_VGAP)
        below = (r.y0 >= caption_rect.y1) and ((r.y0 - caption_rect.y1) <= MAX_VGAP)
        xok   = x_overlap_ratio(r, caption_rect) >= MIN_X_OVERLAP_RATIO
        if xok and (above or below):
            cand.append(r)
    if not cand: return []
    if MERGE_NEAR_IMAGES:
        u = cand[0]
        for r in cand[1:]: u = rect_union(u, r)
        return [u]
    return cand

def render_rect_png_bytes(page: "fitz.Page", rect: "fitz.Rect", scale: float = SCALE) -> bytes:
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    return pix.tobytes("png")

def extract_and_upload_figures(pdf_path: str) -> list[dict]:
    """
    반환: [{id, page, caption_text, upload_id, image_block}, ...]
    """
    if fitz is None:
        return []
    out = []
    out_dir = os.path.join(os.path.dirname(pdf_path), "_figures")
    if SAVE_LOCAL_FIGS:
        os.makedirs(out_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    for pno in range(len(doc)):
        page = doc[pno]
        page_rect = page.rect
        text_blocks = get_text_blocks(page)
        caps = [b for b in text_blocks if b["type"] == 0 and is_caption_text(b["text"])]
        if not caps:
            continue
        vrects = get_visual_rects(page)
        if not vrects:
            continue

        for cidx, cb in enumerate(caps, start=1):
            cap = cb["rect"]
            near = group_for_caption(cap, vrects)
            if not near:
                continue
            uni = cap
            for rr in near:
                uni = rect_union(uni, rr)
            crop = expand_rect(uni, MARGIN, page_rect)
            png_bytes = render_rect_png_bytes(page, crop, SCALE)

            # 옵션: 로컬 저장
            fname = f"p{pno+1:02d}_figure{cidx:02d}.png"
            if SAVE_LOCAL_FIGS:
                with open(os.path.join(out_dir, fname), "wb") as f:
                    f.write(png_bytes)

            # 업로드
            created = create_file_upload(fname, "image/png")
            fid = created.get("id")
            if not fid: 
                continue
            sent = send_file_upload_bytes(fid, png_bytes, fname, "image/png")
            if sent.get("status") != "uploaded":
                continue

            record = {
                "id": f"p{pno+1:02d}_f{cidx:02d}",
                "page": pno+1,
                "caption_text": cb["text"].strip(),
                "upload_id": fid,
                "image_block": image_block_from_upload(fid, caption=""),
            }
            out.append(record)
    doc.close()
    return out


# 9) Build content blocks (with headings + md emphasis)
def build_content_blocks(contents: dict) -> list[dict]:
    """
    섹션별로 Heading1 + 본문(마크다운 강조 반영) 블록을 만든다.
    """
    blocks = []
    for key in SECTION_KEYS:
        body = s(contents.get(key, "")).strip()
        if not body:
            continue
        # Heading1
        blocks.append(_heading(key, 1))
        # 본문: MD → rich_text
        rich = md_to_rich_text(body)
        if rich:
            blocks.append(_paragraph_rich(rich))
    return blocks


# 10) Notion page composer (layout 반영)
def create_notion_page_with_layout(meta: dict,
                                   contents: dict,
                                   layout: list[dict],
                                   figures_by_id: dict,
                                   paper_url: str,
                                   pdf_block: dict,
                                   file_hint: str):
    title_prop = ensure_title_prop_name(DATABASE_ID)
    title_val  = meta.get("title") or os.path.splitext(os.path.basename(file_hint))[0]
    props = {
        title_prop: {"title": [{"text": {"content": s(title_val)}}]},
        "Tag": {"multi_select": to_tag_list(meta.get("tag"))},
        "Authors": {"rich_text": [{"text": {"content": s(meta.get("authors"))}}]},
        "Year": {"rich_text": [{"text": {"content": s(meta.get("year"))}}]},
        "Conference/journal": {"rich_text": [{"text": {"content": s(meta.get("conference_journal"))}}]},
        "One sentence": {"rich_text": [{"text": {"content": s(meta.get('one_sentence',''))}}]},
    }

    # 1) 공통 상단: URL + PDF
    children = []
    if paper_url: children.append(_bookmark(paper_url))
    if pdf_block: children.append(pdf_block)

        # ⬇️ Figure 1 상단 고정
    if PROMOTE_FIRST_FIGURE:
        lead_id = pick_first_figure_id(list(figures_by_id.values()))
        # pick_first_figure_id는 dict가 아니라 list[dict]를 받도록 했으니 약간 수정:
        lead_id = pick_first_figure_id([{"id": fid, **rec} for fid, rec in figures_by_id.items()])
        if lead_id and figures_by_id.get(lead_id):
            children.append(figures_by_id[lead_id]["image_block"])

    # 2) 섞기(섹션/피겨)
    for item in layout:
        if item.get("type") == "section":
            key = item.get("key")
            body = s(contents.get(key, "")).strip()
            if not body:
                continue
            children.append(_heading(key, 1))
            rich = md_to_rich_text(body)
            if rich:
                children.append(_paragraph_rich(rich))
        elif item.get("type") == "figure":
            fid = item.get("id")
            rec = figures_by_id.get(fid)
            if rec:
                children.append(rec["image_block"])

    # 3) 혹시 남은 섹션이 있으면 맨 뒤에 추가
    used_keys = {i.get("key") for i in layout if i.get("type")=="section"}
    for key in SECTION_KEYS:
        if key not in used_keys and s(contents.get(key,"")).strip():
            children.append(_heading(key, 1))
            children.append(_paragraph_rich(md_to_rich_text(contents[key])))

    return notion.pages.create(parent={"database_id": DATABASE_ID}, properties=props, children=children)


# 11) Pipeline
def process_pdf(path: str):
    print(f"📄 Processing: {path}")
    text = read_pdf(path)
    text = strip_unpaired_surrogates(text)
    # (1) 메타
    try:
        meta = call_gpt_meta(text)
    except Exception as e:
        print(f"⚠️ GPT(meta) 오류: {e}")
        meta = {"title":"","authors":"","year":"","conference_journal":"","tag":[]}
    if not s(meta.get("year")):
        g = safe_year_guess(os.path.basename(path), text)
        if g: meta["year"] = g

    # (2) contents (MD 강조 + 정확 키)
    try:
        contents = call_gpt_contents_marked(text)
    except Exception as e:
        print(f"⚠️ GPT(contents) 오류: {e}")
        contents = {k:"" for k in SECTION_KEYS}

    # (3) one-sentence
    try:
        meta["one_sentence"] = s(oai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":"한 문장(30자 이내)만 반환하라."},
                      {"role":"user","content":f"다음 논문을 한글로 30자 이내 한 문장 요약:\n\"\"\"{text}\"\"\""}],
            temperature=0.2, max_tokens=64,
        ).choices[0].message.content).strip().replace("\n"," ")
    except Exception:
        meta["one_sentence"] = ""

    # (4) paper URL
    try:
        paper_url = find_paper_url(meta.get("title",""), meta.get("authors",""), meta.get("year",""))
    except Exception:
        paper_url = ""

    # (5) PDF 업로드
    pdf_block = None
    try:
        created = create_file_upload(os.path.basename(path), "application/pdf")
        fid = created.get("id")
        sent = send_file_upload_path(fid, path, "application/pdf")
        if sent.get("status") == "uploaded":
            pdf_block = file_block_from_upload(fid)
    except Exception as e:
        print(f"⚠️ PDF 업로드 실패: {e}")

    # (6) Figures 추출/업로드 (캡션 보존)
    try:
        figures = extract_and_upload_figures(path)
    except Exception as e:
        print(f"⚠️ Figure 추출/업로드 실패: {e}")
        figures = []

    figures_by_id = {f["id"]: f for f in figures}

    # (7) 레이아웃(섹션+피겨 섞기)
    try:
        layout = call_gpt_layout(contents, figures)
    except Exception as e:
        print(f"⚠️ 레이아웃 생성 오류: {e}")
        layout = [{"type":"section","key":k} for k in SECTION_KEYS] + [{"type":"figure","id":f["id"]} for f in figures]

    # (8) Notion 페이지 생성(레이아웃 반영)
    try:
        res = create_notion_page_with_layout(meta, contents, layout, figures_by_id, paper_url, pdf_block, file_hint=path)
        print(f"✅ Added to Notion: {res.get('url')}\n")
    except APIResponseError as e:
        print(f"❌ Notion API 오류: {e}\n")


# 12) Entry
def main():
    pdf_dir = os.path.join(os.getcwd(), "paper")
    if not os.path.isdir(pdf_dir):
        print(f"❌ 폴더가 없습니다: {pdf_dir}")
        sys.exit(1)

    pdfs = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdfs:
        print("❌ paper_pdf 폴더에 PDF가 없습니다.")
        sys.exit(1)

    for p in pdfs:
        process_pdf(p)

if __name__ == "__main__":
    main()
