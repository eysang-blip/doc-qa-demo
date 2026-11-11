"""PUBG 패치노트 요약 문서 Q&A 데모.

docs/patch_summary.txt를 패치별 섹션으로 나누고, Gemini API로
질문에 답하거나 패치를 3줄로 요약한다. 답변에는 근거 패치 번호를 붙인다.

주의: 입력 문서는 공식 패치노트 원문이 아니라 AI가 작성한 요약본이다.
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

DOC_PATH = Path(__file__).parent / "docs" / "patch_summary.txt"
# gemini-2.5-flash는 신규 사용자 사용 불가, 3.8-flash는 응답이 없어서(2026-11-11 확인) 3.5-flash 사용
MODEL = "gemini-3.5-flash"
TIMEOUT_MS = 60_000

# 예: "1. 업데이트 44.1 (2026.11.03)"
HEADING_RE = re.compile(r"^\s*\d+\.\s*업데이트\s+(\d+\.\d+)\s*\((\d{4}\.\d{2}\.\d{2})\)\s*$", re.M)

NOT_FOUND = "문서에 해당 내용이 없습니다."


def load_patches(path: Path = DOC_PATH) -> list[dict]:
    """문서를 읽어 [{'version', 'date', 'body'}, ...] 형태로 나눈다."""
    text = path.read_text(encoding="utf-8")
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        raise ValueError(f"패치 헤딩을 찾지 못했습니다: {path}")

    patches = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        patches.append({
            "version": m.group(1),
            "date": m.group(2),
            "body": text[m.end():end].strip(),
        })
    return patches


def format_context(patches: list[dict]) -> str:
    return "\n\n".join(
        f"[패치 {p['version']} ({p['date']})]\n{p['body']}" for p in patches
    )


def get_client() -> genai.Client:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY가 .env에 없습니다.")
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))


def _generate(client: genai.Client, system: str, prompt: str) -> str:
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system, temperature=0),
    )
    return (resp.text or "").strip()


PROMPT_VERSION = "v1"  # 프롬프트를 바꾸면 올려서 eval 결과를 비교한다

QA_SYSTEM = f"""당신은 PUBG 패치노트 요약 문서만을 근거로 답하는 도우미입니다.
규칙:
1. 아래 [문서]에 적힌 내용만 사용하세요. 배경지식이나 추측을 더하지 마세요.
2. 답변 끝에 근거가 된 패치를 "(출처: 패치 44.1)"처럼 표시하세요. 여러 개면 모두 적으세요.
3. 문서에 답이 없으면 정확히 "{NOT_FOUND}"라고만 답하세요.
4. 한국어로 간결하게 답하세요."""


def answer(client: genai.Client, patches: list[dict], question: str) -> str:
    prompt = f"[문서]\n{format_context(patches)}\n\n[질문]\n{question}"
    return _generate(client, QA_SYSTEM, prompt)


SUMMARY_SYSTEM = """주어진 PUBG 패치 요약을 정확히 3줄로 다시 요약하세요.
각 줄은 "- "로 시작하고, 문서에 없는 내용은 넣지 마세요."""


def summarize(client: genai.Client, patches: list[dict], version: str) -> str:
    patch = next((p for p in patches if p["version"] == version), None)
    if patch is None:
        versions = ", ".join(p["version"] for p in patches)
        return f"패치 {version}을(를) 찾을 수 없습니다. 가능한 패치: {versions}"
    prompt = f"[패치 {patch['version']} ({patch['date']})]\n{patch['body']}"
    return _generate(client, SUMMARY_SYSTEM, prompt)


def main() -> None:
    patches = load_patches()
    client = get_client()
    print(f"패치 {len(patches)}개 로드: " + ", ".join(p["version"] for p in patches))
    print("질문을 입력하세요. '/요약 44.1' = 3줄 요약, 빈 줄 = 종료\n")

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        try:
            if q.startswith("/요약"):
                print(summarize(client, patches, q.removeprefix("/요약").strip()), "\n")
            else:
                print(answer(client, patches, q), "\n")
        except errors.APIError as e:
            print(f"[API 오류 {e.code}] {e.message}\n잠시 후 다시 시도하세요.\n")
        except Exception as e:  # 타임아웃·네트워크 오류
            print(f"[연결 오류] {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
