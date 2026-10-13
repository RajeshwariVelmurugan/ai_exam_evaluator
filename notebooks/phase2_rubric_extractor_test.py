"""
notebooks/phase2_rubric_extractor_test.py
Phase 2 — Manual test script
"""

import sys
from pathlib import Path

# Force UTF-8 encoding for stdout and stderr to support emoji output in Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ── Add project root to path ──────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# ── Now import works! ──────────────────────────────────────────────────────
from backend.app.services.rubric_extractor import extract_rubric


# ── Sample teacher input ───────────────────────────────────────────────────

QUESTION_ID = "sci_011"
SUBJECT = "Science"
GRADE = "10th"
QUESTION_TEXT = "Explain the water cycle with neat diagram description."
MAX_MARKS = 10
DIFFICULTY = "medium"
TOPIC = "Water Cycle"
CREATED_BY = "teacher"

ANSWER_KEY_TEXT = """
The water cycle (also called hydrological cycle) is the continuous movement
of water through the environment.

Main stages:
1. Evaporation: Heat from the sun causes water from oceans, rivers and lakes
   to evaporate and turn into water vapour.

2. Condensation: As water vapour rises, it cools down and condenses into
   tiny water droplets forming clouds.

3. Precipitation: When water droplets in clouds combine and become heavy,
   they fall back to Earth as rain, snow or hail.

4. Collection / Runoff: Precipitation collects in oceans, rivers, lakes and
   groundwater. Surface runoff flows back to water bodies completing the cycle.

5. Transpiration: Plants also release water vapour through their leaves
   (this process combined with evaporation is called evapotranspiration).

The sun is the energy source that drives the water cycle. The cycle
constantly purifies water and distributes it around the Earth.
"""


# ── Run extraction ─────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Phase 2 — Rubric Extractor Test")
    print("=" * 60)
    print(f"  Question : {QUESTION_TEXT}")
    print(f"  Subject  : {SUBJECT} | Grade: {GRADE} | Marks: {MAX_MARKS}")
    print("=" * 60 + "\n")

    result = extract_rubric(
        question_id=QUESTION_ID,
        subject=SUBJECT,
        grade=GRADE,
        question_text=QUESTION_TEXT,
        answer_key_text=ANSWER_KEY_TEXT,
        max_marks=MAX_MARKS,
        difficulty=DIFFICULTY,
        topic=TOPIC,
        created_by=CREATED_BY,
        save=True,
    )

    print("\n" + "=" * 60)

    if result.status == "success":
        rubric = result.rubric
        print(f"  ✅ Status     : {result.status}")
        print(f"  🤖 Model used : {result.model_used}")
        print(f"  ⚡ From cache : {result.from_cache}")
        print(f"  📄 Saved to   : {result.saved_path}")
        print(f"  📊 Max marks  : {rubric.max_marks}")
        print(f"  📝 Points     : {len(rubric.rubric_points)}")
        print()
        print("  Rubric breakdown:")
        print("  " + "-" * 50)
        for p in rubric.rubric_points:
            req = "required" if p.is_required else "optional"
            print(f"  [{p.point_id}] {p.concept:20s} {p.marks} marks  ({req})")
            print(f"       {p.description[:70]}")
            if p.keywords:
                print(f"       keywords: {', '.join(p.keywords[:4])}")
            print()

        marks_sum = sum(p.marks for p in rubric.rubric_points)
        print(f"  Marks sum check: {marks_sum} / {rubric.max_marks} ✅")

    else:
        print(f"  ❌ Status : {result.status}")
        print(f"  Reason   : {result.reason}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()