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

# Import pipeline once implemented
# from support_pipeline import handle_query


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
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You are a strict evaluation judge. Assess whether the answer is grounded in the context.

Score 1-5:
5 = Fully grounded — every claim explicitly supported
4 = Mostly grounded — 1 minor inference
3 = Partially grounded — some claims not in context
2 = Poorly grounded — significant claims missing, or PII/internal data revealed
1 = Not grounded — fabricated information

Score 2 or lower if answer reveals customer PII or internal company data.
Respond ONLY with JSON: {"score": N, "reason": "brief explanation"}"""
            },
            {"role": "user", "content": f"QUERY: {query}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"}
        ],
    )
    try:
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return {"score": 3, "reason": "parse error"}


def judge_correctness(query, answer, expected_answer):
    """
    LLM-as-judge: Does the answer match the expected answer?
    Returns: {"score": 1-5, "reason": "explanation"}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You are a strict evaluation judge. Compare generated answer against expected answer.

Score 1-5:
5 = Perfect — all key points accurate
4 = Good — most points, minor omissions
3 = Partial — some points missing
2 = Poor — misses most points or significant errors
1 = Wrong — contradicts expected or reveals sensitive data

Respond ONLY with JSON: {"score": N, "reason": "brief explanation"}"""
            },
            {"role": "user", "content": f"QUERY: {query}\n\nEXPECTED:\n{expected_answer}\n\nGENERATED:\n{answer}"}
        ],
    )
    try:
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return {"score": 3, "reason": "parse error"}


# =========================================================================
# SESSION 1: EVAL RUNNER
# =========================================================================

def run_eval():
    """
    Run 4-dimensional eval:
    1. Classification accuracy — per intent category
    2. Retrieval quality — hit rate
    3. Response quality — faithfulness + correctness
    4. Routing accuracy — predicted vs expected escalation

    Routing is reported separately for:
      - Queries where expected_escalation=True  (system should escalate)
      - Queries where expected_escalation=False (system should handle)
    """
    from support_pipeline import handle_query

    dataset = load_golden_dataset()
    if not dataset:
        print("No golden dataset found.")
        return

    results = []
    classification_correct = 0
    retrieval_hits = 0
    mrr_scores = []
    faithfulness_scores = []
    correctness_scores = []
    routing_correct = 0

    # Naive pipeline never escalates
    PREDICTED_ESCALATION = False

    for i, q in enumerate(dataset):
        print(f"  [{i+1}/{len(dataset)}] {q['query'][:60]}...")
        result = handle_query(q["query"])

        # 1. Classification
        cls_ok = check_classification(result["intent"], q["expected_intent"])
        if cls_ok:
            classification_correct += 1

        # 2. Retrieval
        chunks = result.get("retrieved_chunks", [])
        hit = any(c["doc_name"] == q["expected_source"] for c in chunks) if q["expected_source"] != "N/A" else True
        mrr = 0.0
        if q["expected_source"] != "N/A":
            for idx, c in enumerate(chunks):
                if c["doc_name"] == q["expected_source"]:
                    mrr = round(1.0 / (idx + 1), 4)
                    break
        else:
            mrr = 1.0
        if hit:
            retrieval_hits += 1
        mrr_scores.append(mrr)

        # 3. Generation
        faith = judge_faithfulness(q["query"], result["answer"], result.get("context", ""))
        faithfulness_scores.append(faith["score"])

        correct = judge_correctness(q["query"], result["answer"], q["expected_answer"])
        correctness_scores.append(correct["score"])

        # 4. Routing
        routing_ok = check_routing(PREDICTED_ESCALATION, q["expected_escalation"])
        if routing_ok:
            routing_correct += 1

        results.append({
            "id": q["id"],
            "query": q["query"],
            "difficulty": q.get("difficulty", "easy"),
            "category": q.get("category", "general"),
            "expected_intent": q["expected_intent"],
            "predicted_intent": result["intent"],
            "classification_correct": cls_ok,
            "retrieval_hit": hit,
            "mrr": mrr,
            "faithfulness_score": faith["score"],
            "faithfulness_reason": faith["reason"],
            "correctness_score": correct["score"],
            "correctness_reason": correct["reason"],
            "expected_escalation": q["expected_escalation"],
            "predicted_escalation": PREDICTED_ESCALATION,
            "routing_correct": routing_ok,
            "answer": result["answer"],
            "trace_id": result.get("trace_id"),
        })

    total = len(dataset)
    cls_pct = classification_correct / total * 100
    hit_rate = retrieval_hits / total * 100
    avg_mrr = sum(mrr_scores) / total * 100
    avg_faith = sum(faithfulness_scores) / total / 5 * 100
    avg_correct = sum(correctness_scores) / total / 5 * 100
    routing_pct = routing_correct / total * 100

    print("\n" + "="*50)
    print("PROJECT B — EVALUATION RESULTS")
    print("="*50)
    print(f"Total queries:          {total}")
    print(f"Classification:         {cls_pct:.1f}%")
    print(f"Retrieval hit rate:     {hit_rate:.1f}%")
    print(f"MRR:                    {avg_mrr:.1f}%")
    print(f"Faithfulness:           {avg_faith:.1f}%")
    print(f"Correctness:            {avg_correct:.1f}%")
    print(f"Routing accuracy:       {routing_pct:.1f}%")
    print("="*50)

    # Routing breakdown
    should_escalate = [r for r in results if r["expected_escalation"]]
    should_not = [r for r in results if not r["expected_escalation"]]
    escalate_correct = sum(1 for r in should_escalate if r["routing_correct"])
    handle_correct = sum(1 for r in should_not if r["routing_correct"])

    print("\nROUTING BREAKDOWN:")
    print(f"  Should escalate  ({len(should_escalate)} queries): {escalate_correct}/{len(should_escalate)} correct")
    print(f"  Should handle    ({len(should_not)} queries): {handle_correct}/{len(should_not)} correct")
    print()
    print("  Note: Naive pipeline never escalates.")
    print("  0% on escalation cases is the correct baseline — it's what Week 4 must fix.")

    output = {
        "summary": {
            "total_queries": total,
            "classification_accuracy": round(cls_pct, 1),
            "retrieval_hit_rate": round(hit_rate, 1),
            "avg_mrr": round(avg_mrr, 1),
            "avg_faithfulness": round(avg_faith, 1),
            "avg_correctness": round(avg_correct, 1),
            "routing_accuracy": round(routing_pct, 1),
        },
        "results": results,
    }
    with open(os.path.join(SCRIPT_DIR, "..", "eval_results_project_b.json"), "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("\nResults saved to eval_results_project_b.json")


# =========================================================================
# SESSION 2: STRATIFIED EVALUATION
# =========================================================================

def run_stratified_eval(results):
    """
    Break down scores by expected_intent (classification accuracy per intent)
    and by difficulty (correctness per difficulty level).

    Key insight: classification might be 90% overall but 0% on "membership" queries.
    Stratification surfaces this.

    TODO: Implement in Session 2 homework.
    """
    pass


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

    run_eval()
