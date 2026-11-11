"""평가 스크립트: 테스트 질문을 돌려 결과를 SQLite(qa_log.db)에 저장하고 집계한다.

사용법:
    python eval.py                     # 전체 평가 실행 후 리포트 출력
    python eval.py --note "프롬프트 수정"  # 실행에 메모 남기기
    python eval.py --limit 3           # 앞의 3문항만 (빠른 확인용)
    python eval.py --report            # 실행 없이 저장된 결과만 집계

채점 방식 (자동, 규칙 기반):
    - 답이 있는 질문: 정답 키워드가 모두 들어 있고, 기대 출처 패치를 모두 인용해야 정답
    - 문서에 없는 질문: "문서에 해당 내용이 없습니다"라고 답해야 정답
"""

import argparse
import json
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from google.genai import errors

import main

ROOT = Path(__file__).parent
DB_PATH = ROOT / "qa_log.db"
QUESTIONS_PATH = ROOT / "eval_questions.json"
REQUEST_GAP_SEC = 4  # 무료 티어 분당 한도(RPM)에 걸리지 않도록 요청 사이 대기

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT NOT NULL,
    model           TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    note            TEXT
);
CREATE TABLE IF NOT EXISTS results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES runs(run_id),
    qid          INTEGER NOT NULL,
    category     TEXT NOT NULL,
    question     TEXT NOT NULL,
    expected     TEXT NOT NULL,
    answer       TEXT,
    correct      INTEGER NOT NULL,      -- 1 = 정답, 0 = 오답
    fail_reason  TEXT,                  -- 오답일 때 이유
    latency_ms   INTEGER
);
"""

NOT_FOUND_MARK = "문서에 해당 내용이 없"
SOURCE_RE = re.compile(r"출처[^)\n]*")
VERSION_RE = re.compile(r"\d{2}\.\d")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def cited_versions(answer: str) -> set[str]:
    """답변의 '(출처: 패치 44.1, 패치 43.1)' 부분에서 패치 번호를 뽑는다."""
    found = set()
    for part in SOURCE_RE.findall(answer):
        found.update(VERSION_RE.findall(part))
    return found


def grade(q: dict, answer: str) -> tuple[bool, str | None]:
    """(정답 여부, 실패 이유)를 돌려준다."""
    refused = NOT_FOUND_MARK in answer

    if q.get("expect_not_found"):
        if refused:
            return True, None
        return False, "환각: 문서에 없는 내용을 답함"

    if refused:
        return False, "과잉 거절: 문서에 있는데 없다고 함"

    normalized = answer.replace(" ", "")
    missing = [k for k in q["keywords"] if k.replace(" ", "") not in normalized]
    if missing:
        return False, f"키워드 누락: {', '.join(missing)}"

    missing_src = set(q["sources"]) - cited_versions(answer)
    if missing_src:
        return False, f"출처 누락: {', '.join(sorted(missing_src))}"

    return True, None


def ask_with_retry(client, patches, question: str, tries: int = 3) -> str:
    """503(과부하)·429(한도 초과)는 잠시 기다렸다가 다시 시도한다."""
    for attempt in range(tries):
        try:
            return main.answer(client, patches, question)
        except errors.APIError as e:
            if e.code not in (429, 503) or attempt == tries - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"    API {e.code}, {wait}초 후 재시도...")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def run_eval(note: str | None, limit: int | None) -> int:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if limit:
        questions = questions[:limit]

    patches = main.load_patches()
    client = main.get_client()
    conn = connect()

    cur = conn.execute(
        "INSERT INTO runs (started_at, model, prompt_version, note) VALUES (?, ?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), main.MODEL, main.PROMPT_VERSION, note),
    )
    run_id = cur.lastrowid
    conn.commit()
    print(f"[run {run_id}] model={main.MODEL} prompt={main.PROMPT_VERSION} 문항 {len(questions)}개\n")

    for q in questions:
        expected = "문서에 없음" if q.get("expect_not_found") else (
            f"키워드 {q['keywords']} / 출처 {q['sources']}"
        )
        start = time.perf_counter()
        try:
            answer = ask_with_retry(client, patches, q["question"])
            ok, reason = grade(q, answer)
        except Exception as e:  # API 오류도 결과로 남긴다
            answer, ok, reason = None, False, f"API 오류: {type(e).__name__}"
        latency = int((time.perf_counter() - start) * 1000)

        conn.execute(
            """INSERT INTO results
               (run_id, qid, category, question, expected, answer, correct, fail_reason, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, q["id"], q["category"], q["question"], expected, answer, int(ok), reason, latency),
        )
        conn.commit()
        mark = "O" if ok else "X"
        print(f"  {mark} Q{q['id']:>2} {q['question']}" + ("" if ok else f"  → {reason}"))
        time.sleep(REQUEST_GAP_SEC)

    conn.close()
    return run_id


def report() -> None:
    conn = connect()

    print("\n=== 실행별 정답률 ===")
    rows = conn.execute("""
        SELECT r.run_id, r.started_at, r.model, r.prompt_version,
               COUNT(*) AS total, SUM(s.correct) AS ok,
               ROUND(100.0 * AVG(s.correct), 1) AS acc,
               ROUND(AVG(s.latency_ms) / 1000.0, 1) AS avg_sec,
               COALESCE(r.note, '')
        FROM runs r JOIN results s ON s.run_id = r.run_id
        GROUP BY r.run_id ORDER BY r.run_id
    """).fetchall()
    if not rows:
        print("저장된 결과가 없습니다. 먼저 python eval.py 를 실행하세요.")
        return
    for run_id, at, model, pv, total, ok, acc, sec, note in rows:
        print(f"  run {run_id} | {at} | {model} | {pv} | {ok}/{total} ({acc}%) | 평균 {sec}s | {note}")

    last = rows[-1][0]
    print(f"\n=== run {last} 유형별 정답률 ===")
    for cat, total, ok in conn.execute("""
        SELECT category, COUNT(*), SUM(correct) FROM results
        WHERE run_id = ? GROUP BY category ORDER BY category
    """, (last,)):
        print(f"  {cat}: {ok}/{total}")

    print(f"\n=== run {last} 오답 목록 ===")
    fails = conn.execute("""
        SELECT qid, question, fail_reason, answer FROM results
        WHERE run_id = ? AND correct = 0 ORDER BY qid
    """, (last,)).fetchall()
    if not fails:
        print("  없음")
    for qid, question, reason, answer in fails:
        short = (answer or "").replace("\n", " ")[:120]
        print(f"  Q{qid} {question}\n      이유: {reason}\n      답변: {short}")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="문서 Q&A 평가")
    parser.add_argument("--note", help="이번 실행에 남길 메모")
    parser.add_argument("--limit", type=int, help="앞에서부터 N문항만 실행")
    parser.add_argument("--report", action="store_true", help="실행 없이 리포트만 출력")
    args = parser.parse_args()

    if not args.report:
        run_eval(args.note, args.limit)
    report()
