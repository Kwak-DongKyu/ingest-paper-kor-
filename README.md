# Ingest Papers to Notion (PDF → GPT Summary → Notion)

논문 PDF를 넣으면:
1) GPT로 메타데이터/요약/한줄요약 생성  
2) PDF/피겨(캡션 포함) 추출 후 **Notion 페이지**에 업로드  
3) 논문 URL(doi/arXiv/semantic scholar) 북마크 + 섹션/피겨 **자동 배치**

## Demo
- 입력: `paper_pdf/` 폴더 안의 PDF들
- 출력: Notion DB의 새 페이지(상단: 논문 URL → PDF → Figure 1 → 섹션/피겨 섞기)

---

## 1) 요구사항
- Python 3.10+
- Notion 계정 (데이터베이스 + 내부 통합)
- OpenAI API 키

---

## 2) 설치

```bash
git clone https://github.com/<yourname>/ingest-papers.git
cd ingest-papers

# (권장) 가상환경
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
