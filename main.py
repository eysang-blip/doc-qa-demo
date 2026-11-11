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
# 2026-11-11 확인: 2.5-flash는 신규 사용자 사용 불가(404), 3.8-flash는 응답 없음,
# 3.5-flash는 무료 티어 하루 20회 제한이라 평가(20문항 x 여러 번)에 부족 → flash-lite 사용
MODEL = "gemini-3.5-flash-lite"
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


# 프롬프트 버전별로 보관해서 eval.py --prompt v1 / v2 로 전후 비교한다
QA_PROMPTS = {
    "v1": f"""당신은 PUBG 패치노트 요약 문서만을 근거로 답하는 도우미입니다.
규칙:
1. 아래 [문서]에 적힌 내용만 사용하세요. 배경지식이나 추측을 더하지 마세요.
2. 답변 끝에 근거가 된 패치를 "(출처: 패치 44.1)"처럼 표시하세요. 여러 개면 모두 적으세요.
3. 문서에 답이 없으면 정확히 "{NOT_FOUND}"라고만 답하세요.
4. 한국어로 간결하게 답하세요.""",

    # v2: v1 평가에서 잘못된 전제(틀린 패치 번호) 질문을 통째로 거절하는 문제 수정
    "v2": f"""당신은 PUBG 패치노트 요약 문서만을 근거로 답하는 도우미입니다.
규칙:
1. 아래 [문서]에 적힌 내용만 사용하세요. 배경지식이나 추측을 더하지 마세요.
2. 답변 끝에 근거가 된 패치를 "(출처: 패치 44.1)"처럼 표시하세요. 여러 개면 모두 적으세요.
3. 질문 속 패치 번호나 전제가 문서와 다르면, 먼저 바로잡고 문서에 있는 내용으로 답하세요.
   예: "RPD는 43.2가 아니라 42.3에서 추가되었습니다. 피해량은 43입니다. (출처: 패치 42.3)"
   단, 질문이 묻는 대상 자체가 문서에 없으면 다른 패치 내용으로 바꿔 답하지 마세요.
4. 질문의 일부만 문서에 있으면 있는 부분만 답하고, 나머지는 "그 외 내용은 문서에 없습니다"라고 덧붙이세요.
5. 관련 내용이 문서에 전혀 없을 때만 정확히 "{NOT_FOUND}"라고만 답하세요.
6. 한국어로 간결하게 답하세요.""",

    # v3: v2 실패 분석 반영
    #  - v2 예시 문장이 평가 문항(Q20) 정답과 같아 평가가 오염됨 → 구체적 예시 제거
    #  - 전제 정정을 과하게 적용해 다른 패치 내용으로 바꿔 답함(Q17, Q19) → 적용 범위 한정
    #  - 무기군(LMG) 단위 변경을 개별 무기(M249) 질문에서 누락(Q21) → 포함 규칙 추가
    "v3": f"""당신은 PUBG 패치노트 요약 문서만을 근거로 답하는 도우미입니다.
규칙:
1. 아래 [문서]에 적힌 내용만 사용하세요. 배경지식이나 추측을 더하지 마세요.
2. 답변 끝에 근거가 된 패치를 "(출처: 패치 번호)" 형식으로 표시하세요. 여러 개면 모두 적으세요.
3. 질문이 특정 무기·기능의 이름을 대면서 패치 번호만 틀리게 말했다면, 올바른 패치 번호로 바로잡은 뒤 답하세요.
4. 질문이 "어느 패치의 보상", "어느 패치의 새 총기"처럼 특정 패치 안의 내용을 물으면 그 패치 안에서만 찾으세요.
   그 패치에 없으면 다른 패치의 내용으로 대신 답하지 말고 거절하세요.
5. 개별 무기에 대한 질문에는 그 무기가 속한 무기군(예: LMG) 전체에 적용된 변경도 포함하세요.
6. 수량의 증감을 물으면 변경 전 값, 변경 후 값, 그 차이를 함께 답하세요.
7. 질문이 묻는 정보(기간, 수치 등)가 문서에 없으면 관련된 다른 사실을 덧붙이지 말고, 정확히 "{NOT_FOUND}"라고만 답하세요.
8. 한국어로 간결하게 답하세요.""",
}
PROMPT_VERSION = "v3"  # 기본값 (main.py 대화형 실행에 사용)


def answer(client: genai.Client, patches: list[dict], question: str,
           version: str = PROMPT_VERSION) -> str:
    prompt = f"[문서]\n{format_context(patches)}\n\n[질문]\n{question}"
    return _generate(client, QA_PROMPTS[version], prompt)


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
