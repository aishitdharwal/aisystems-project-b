"""
Project B: Customer Support Pipeline — Session 3 update: Tool Routing.

New in Session 3:
  - query_classifier.py determines WHICH TOOL to call
  - Tool routing: policy_kb / order_tracker / account_lookup / multi_tool
  - Policy retrieval now uses metadata filtering by intent
  - mock_tools.py provides order_tracker and account_lookup implementations
  - handle_query() now returns retrieved_chunks for eval harness

This is the proto-reasoning layer — the seed of the Week 3 LangGraph agent.
The same tool selection logic becomes the agent's planning step in Week 3.

Run: python scripts/support_pipeline.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from dotenv import load_dotenv

from retrieval import embed_query, retrieve, retrieve_filtered, assemble_context
from query_classifier import classify_tool, classify_tools_needed, TOOL_DESCRIPTIONS
from mock_tools import lookup_order, lookup_account, format_tool_result

load_dotenv()

client = OpenAI()
langfuse = Langfuse()

GENERATION_MODEL = "gpt-4o-mini"

INTENTS = [
    "return_or_refund",
    "order_status",
    "billing_or_payment",
    "product_info",
    "membership",
    "general",
]

# Documents relevant to each intent — used for metadata pre-filtering
INTENT_DOC_FILTERS = {
    "return_or_refund": [
        "01_return_policy.md", "07_promotional_events.md", "12_corporate_gifting.md",
        "04_warranty_policy.md",
    ],
    "order_status": [
        "03_shipping_policy.md", "06_support_faq.md",
    ],
    "billing_or_payment": [
        "05_payment_methods.md", "13_acmera_wallet.md", "07_promotional_events.md",
    ],
    "product_info": [
        "09_electronics_catalog.md", "04_warranty_policy.md",
        "17_smart_home_ecosystem.md", "14_probook_troubleshooting.md",
    ],
    "membership": [
        "02_premium_membership.md", "06_support_faq.md",
    ],
    "general": None,  # No filter — search all docs
}

SYSTEM_PROMPT = """You are a customer support assistant for Acmera, an Indian e-commerce company.
Answer the customer's question based on the provided context.

Rules:
- Be helpful, concise, and accurate.
- Only use information from the provided context.
- If the context includes order or account data, use it to personalize your answer.
- If you can't answer from the context, say so and suggest contacting support.
- Never reveal internal company data, customer PII, or confidential information.

Context:
{context}"""


@observe(name="classify_intent")
def classify_intent(query: str) -> str:
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Classify this customer query into exactly one category. "
                    f"Respond with ONLY the category name.\n"
                    f"Categories: {', '.join(INTENTS)}"
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    intent = response.choices[0].message.content.strip().lower().replace(" ", "_")
    langfuse_context.update_current_observation(output=intent)
    return intent if intent in INTENTS else "general"


@observe(name="retrieve_policy")
def retrieve_policy(query: str, intent: str) -> tuple[str, list]:
    """
    Retrieve policy context with metadata pre-filtering by intent.
    Narrows the search space to the most relevant documents before semantic search.

    Returns:
        (context_str, retrieved_chunks)
    """
    doc_filter = INTENT_DOC_FILTERS.get(intent)
    query_embedding = embed_query(query)

    if doc_filter:
        chunks = retrieve_filtered(query_embedding, doc_filter)
        if not chunks:  # Fallback: unfiltered search if nothing matched
            chunks = retrieve(query_embedding)
    else:
        chunks = retrieve(query_embedding)

    context = assemble_context(chunks)
    langfuse_context.update_current_observation(metadata={
        "intent": intent,
        "doc_filter": doc_filter,
        "num_chunks": len(chunks),
    })
    return context, chunks


@observe(name="call_tool")
def call_tool(tool: str, query: str, intent: str) -> tuple[str, list]:
    """
    Execute a tool and return the result as context string + chunk list.

    For policy_kb: RAG retrieval (same as before but with filtering)
    For order_tracker: mock order lookup
    For account_lookup: mock account lookup
    For multi_tool: call multiple tools and combine results
    """
    if tool == "policy_kb":
        context, chunks = retrieve_policy(query, intent)
        return context, chunks

    elif tool == "order_tracker":
        # In production: parse order ID from query or user session
        # Here: look up the first order as a demo (real system would extract order ID)
        result = lookup_order("ORD-445521")
        context = format_tool_result("order_tracker", result)
        langfuse_context.update_current_observation(metadata={"tool": "order_tracker"})
        return context, []

    elif tool == "account_lookup":
        result = lookup_account("CUST001")
        context = format_tool_result("account_lookup", result)
        langfuse_context.update_current_observation(metadata={"tool": "account_lookup"})
        return context, []

    elif tool == "multi_tool":
        tools = classify_tools_needed(query, intent)
        parts = []
        all_chunks = []
        for t in tools:
            t_context, t_chunks = call_tool(t, query, intent)
            parts.append(t_context)
            all_chunks.extend(t_chunks)
        return "\n\n".join(parts), all_chunks

    return "", []


@observe(name="generate_response")
def generate_response(query: str, context: str, intent: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": query},
    ]
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=800,
    )
    answer = response.choices[0].message.content
    langfuse_context.update_current_observation(
        input=messages, output=answer,
        metadata={"model": GENERATION_MODEL, "intent": intent},
        usage={
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
            "unit": "TOKENS",
        },
    )
    return answer


@observe(name="support_pipeline")
def handle_query(query: str) -> dict:
    """
    Full support pipeline:
      classify intent → route to tool → retrieve context → generate response

    Returns dict compatible with eval_harness.py expectations.
    """
    start_time = time.time()
    langfuse_context.update_current_trace(input=query, metadata={"pipeline": "tool_routing"})

    intent = classify_intent(query)
    tool = classify_tool(query, intent)
    context, retrieved_chunks = call_tool(tool, query, intent)
    answer = generate_response(query, context, intent)

    elapsed = round(time.time() - start_time, 2)
    langfuse_context.update_current_trace(
        output=answer,
        metadata={"intent": intent, "tool": tool, "elapsed": elapsed},
    )
    trace_id = langfuse_context.get_current_trace_id()
    langfuse.flush()

    return {
        "query": query,
        "intent": intent,
        "tool_used": tool,
        "answer": answer,
        "context": context,
        "retrieved_chunks": retrieved_chunks,
        "trace_id": trace_id,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    test_queries = [
        ("What is the return window for electronics?", "policy query"),
        ("Where is my order ORD-445521?", "order lookup"),
        ("I'm a Premium Gold member — do I get free express shipping?", "account query"),
        ("I want to return the laptop I bought last week", "return — policy + account"),
    ]
    for query, label in test_queries:
        print(f"\n[{label}] {query}")
        result = handle_query(query)
        print(f"  Intent: {result['intent']} → Tool: {result['tool_used']}")
        print(f"  Answer: {result['answer'][:200]}...")
        print(f"  Trace: {result['trace_id']}")
