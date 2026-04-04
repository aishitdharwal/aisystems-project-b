"""
Project B Evaluation Harness — Sessions 1 & 2 Starter

4-dimensional eval for the support pipeline:

SESSION 1 functions (implement during Session 1 homework):
  1. check_classification() — did it identify the right intent?
  2. check_routing() — should this have been escalated?
  3. judge_faithfulness() — is the answer grounded in context?
  4. judge_correctness() — does it match the expected answer?
  5. run_eval() — orchestrate and produce scorecard

SESSION 2 functions (implement during Session 2 homework):
  6. run_stratified_eval() — break down by intent and difficulty
  7. attach_langfuse_scores() — attach all 4 dimensions to LangFuse traces
  8. save_baseline() — lock current scores as regression anchor

Run: python scripts/eval_harness.py
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

SCRIPT_DIR = os.path.dirname(__file__)

# Pipeline imported inside run_eval() to avoid import-time side effects


# =========================================================================
# GOLDEN DATASET
# =========================================================================

def load_golden_dataset():
    """Load Project B's golden dataset."""
    path = os.path.join(SCRIPT_DIR, "golden_dataset.json")
    if not os.path.exists(path):
        print("No golden_dataset.json found for Project B.")
        return []
    with open(path) as f:
        return json.load(f)


# =========================================================================
# SESSION 1: CLASSIFICATION METRICS
# =========================================================================

def check_classification(predicted_intent, expected_intent):
    """
    Did the system classify the query correctly?
    Returns True/False.

    Hint: direct string equality check.

    TODO: Implement in Session 1 homework.
    """
    return predicted_intent == expected_intent


# =========================================================================
# SESSION 1: ROUTING METRICS
# =========================================================================

def check_routing(predicted_escalation, expected_escalation):
    """
    Should this query have been escalated to a human?
    Did the system make the right routing decision?
    Returns True/False.

    Note: the naive pipeline NEVER escalates (always False).
    So this will be True only for queries where expected_escalation=False.
    That 0% score on escalation cases IS the correct baseline — it's what Week 4 fixes.
    """
    return predicted_escalation == expected_escalation


# =========================================================================
# SESSION 1: GENERATION METRICS
# =========================================================================

def judge_faithfulness(query, answer, context):
    """
    LLM-as-judge: Is the answer grounded in context?
    Returns: {"score": 1-5, "reason": "explanation"}

    Same pattern as Project A — identical judge prompt.
    """
    prompt = f"""You are an evaluation judge. Score whether the answer is grounded in the provided context.

Rubric:
- Score 5: Every claim in the answer is explicitly supported by the context.
- Score 4: Almost all claims supported; minor unsupported details.
- Score 3: Some claims are supported but others are not in the context.
- Score 2: Most claims are not supported by the context.
- Score 1: Answer contains fabricated information not present in the context.

Question: {query}

Context:
{context}

Answer:
{answer}

Respond with JSON only, no markdown fences:
{{"score": <1-5>, "reason": "<one sentence explanation>"}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def judge_correctness(query, answer, expected_answer):
    """
    LLM-as-judge: Does the answer match the expected answer?
    Returns: {"score": 1-5, "reason": "explanation"}
    """
    prompt = f"""You are an evaluation judge. Score whether the generated answer correctly addresses the question compared to the expected answer.

Rubric:
- Score 5: Generated answer is fully correct and covers all key points of the expected answer.
- Score 4: Mostly correct with minor omissions or imprecise details.
- Score 3: Partially correct — captures some key points but misses others.
- Score 2: Mostly incorrect or significantly incomplete.
- Score 1: Wrong or completely unrelated to the expected answer.

Question: {query}

Expected answer: {expected_answer}

Generated answer: {answer}

Respond with JSON only, no markdown fences:
{{"score": <1-5>, "reason": "<one sentence explanation>"}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


# =========================================================================
# SESSION 1: EVAL RUNNER
# =========================================================================

def run_eval():
    """
    Run 4-dimensional eval:
    1. Classification accuracy — predicted vs expected intent
    2. Routing accuracy — broken down by should-escalate vs should-handle
    3. Faithfulness — answer grounded in retrieved context
    4. Correctness — answer matches expected answer

    Naive pipeline: predicted_escalation is always False.
    Routing on should-escalate cases = 0% (Week 4 baseline).
    """
    from support_pipeline import handle_query
    from retrieval import embed_query, retrieve, assemble_context

    dataset = load_golden_dataset()
    if not dataset:
        return

    print(f"\nRunning Project B eval on {len(dataset)} queries...\n")

    results = []
    for entry in dataset:
        print(f"  [{entry['id']}] {entry['query'][:60]}...")

        pipeline_result = handle_query(entry["query"])

        # Naive pipeline never escalates
        predicted_escalation = False

        # Retrieve context independently for faithfulness scoring
        query_embedding = embed_query(entry["query"])
        chunks = retrieve(query_embedding)
        context = assemble_context(chunks)

        classification_correct = check_classification(pipeline_result["intent"], entry["expected_intent"])
        routing_correct = check_routing(predicted_escalation, entry["expected_escalation"])
        faith = judge_faithfulness(entry["query"], pipeline_result["answer"], context)
        correct = judge_correctness(entry["query"], pipeline_result["answer"], entry["expected_answer"])

        print(f"         class={classification_correct}  routing={routing_correct}"
              f"  faith={faith['score']}  correct={correct['score']}")

        results.append({
            "id": entry["id"],
            "query": entry["query"],
            "category": entry.get("category", "unknown"),
            "difficulty": entry.get("difficulty", "easy"),
            "expected_intent": entry["expected_intent"],
            "predicted_intent": pipeline_result["intent"],
            "expected_escalation": entry["expected_escalation"],
            "predicted_escalation": predicted_escalation,
            "expected_answer": entry["expected_answer"],
            "answer": pipeline_result["answer"],
            "trace_id": pipeline_result["trace_id"],
            "elapsed_seconds": pipeline_result["elapsed_seconds"],
            "classification_correct": classification_correct,
            "routing_correct": routing_correct,
            "faithfulness": faith,
            "correctness": correct,
        })

    # --- Scorecard ---
    n = len(results)

    # Dimension 1: Classification
    classification_acc = sum(1 for r in results if r["classification_correct"]) / n

    # Dimension 2: Routing — broken down
    should_escalate = [r for r in results if r["expected_escalation"]]
    should_handle   = [r for r in results if not r["expected_escalation"]]

    missed_escalation_caught = (
        sum(1 for r in should_escalate if r["routing_correct"]) / len(should_escalate)
        if should_escalate else float("nan")
    )
    correct_handle = (
        sum(1 for r in should_handle if r["routing_correct"]) / len(should_handle)
        if should_handle else float("nan")
    )

    # Dimension 3 & 4: Generation
    avg_faith   = sum(r["faithfulness"]["score"] for r in results) / n
    avg_correct = sum(r["correctness"]["score"] for r in results) / n

    print()
    print("=" * 52)
    print("  PROJECT B SCORECARD")
    print("=" * 52)
    print(f"  Queries evaluated     : {n}")
    print()
    print(f"  [1] Classification accuracy : {classification_acc:.0%}")
    print()
    print(f"  [2] Routing accuracy")
    print(f"      correct-handle          : {correct_handle:.0%}"
          f"  ({len(should_handle)} queries where expected_escalation=False)")
    print(f"      missed-escalation caught: {missed_escalation_caught:.0%}"
          f"  ({len(should_escalate)} queries where expected_escalation=True)")
    print(f"      → 0% missed-escalation is the Week 4 starting point")
    print()
    print(f"  [3] Avg faithfulness        : {avg_faith:.2f} / 5")
    print(f"  [4] Avg correctness         : {avg_correct:.2f} / 5")
    print("=" * 52)

    # --- Save results ---
    out_path = os.path.join(SCRIPT_DIR, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


# =========================================================================
# SESSION 2: STRATIFIED EVALUATION
# =========================================================================

def run_stratified_eval(results):
    """
    Break down scores by expected_intent (classification accuracy per intent)
    and by difficulty (correctness per difficulty level).

    Key insight: classification might be 90% overall but 0% on "membership" queries.
    Stratification surfaces this.
    """
    from collections import defaultdict

    # --- Per-intent classification accuracy ---
    intent_buckets = defaultdict(list)
    for r in results:
        intent_buckets[r["expected_intent"]].append(r)

    print()
    print("=" * 60)
    print("  STRATIFIED EVAL — Classification accuracy by intent")
    print("=" * 60)
    print(f"  {'Intent':<22} {'Total':>5}  {'Correct':>7}  {'Accuracy':>8}  Misclassified as")
    print(f"  {'-'*22}  {'-'*5}  {'-'*7}  {'-'*8}  ----------------")

    intent_rows = []
    for intent, rows in sorted(intent_buckets.items()):
        total = len(rows)
        correct = sum(1 for r in rows if r["classification_correct"])
        accuracy = correct / total
        wrong = [r["predicted_intent"] for r in rows if not r["classification_correct"]]
        wrong_str = ", ".join(wrong) if wrong else "—"
        intent_rows.append((accuracy, intent, total, correct, wrong_str))

    # Print worst-first
    for accuracy, intent, total, correct, wrong_str in sorted(intent_rows):
        flag = " ◄ worst" if accuracy == min(a for a, *_ in intent_rows) else ""
        print(f"  {intent:<22} {total:>5}  {correct:>7}  {accuracy:>7.0%}  {wrong_str}{flag}")

    print()
    worst_intent = min(intent_rows, key=lambda x: x[0])
    print(f"  Worst intent: '{worst_intent[1]}' at {worst_intent[0]:.0%} accuracy")

    # --- Per-difficulty correctness ---
    diff_buckets = defaultdict(list)
    for r in results:
        diff_buckets[r["difficulty"]].append(r)

    print()
    print("=" * 60)
    print("  STRATIFIED EVAL — Avg correctness by difficulty")
    print("=" * 60)
    print(f"  {'Difficulty':<12} {'Total':>5}  {'Avg Correct':>11}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*11}")
    for diff in ["easy", "medium", "hard"]:
        if diff not in diff_buckets:
            continue
        rows = diff_buckets[diff]
        avg = sum(r["correctness"]["score"] for r in rows) / len(rows)
        print(f"  {diff:<12} {len(rows):>5}  {avg:>10.2f}/5")

    print("=" * 60)

    return intent_rows


# =========================================================================
# SESSION 2: LANGFUSE SCORE ATTACHMENT
# =========================================================================

def attach_langfuse_scores(trace_id, classification_correct, retrieval_hit,
                            faithfulness_result, correctness_result, routing_correct):
    """
    Attach all 4 eval dimensions to a LangFuse trace.

    Scores to attach:
      - "classification_correct": 1.0 or 0.0
      - "retrieval_hit": 1.0 or 0.0
      - "faithfulness": faithfulness_result["score"] / 5
      - "correctness": correctness_result["score"] / 5
      - "routing_correct": 1.0 or 0.0

    TODO: Implement in Session 2 homework.
    """
    pass


# =========================================================================
# SESSION 2: SAVE BASELINE
# =========================================================================

def save_baseline(summary_scores):
    """
    Save current Project B scores as baseline_scores.json.
    Include all 4 dimensions in the baseline.

    TODO: Implement in Session 2 homework.
    """
    pass


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--category", type=str)
    args = parser.parse_args()

    results = run_eval()
    if results:
        run_stratified_eval(results)
