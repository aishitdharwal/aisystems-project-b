"""
Self-contained retrieval layer for Project B — Session 4 update.

New in Session 4:
  - deduplicate_chunks()  — removes near-identical chunks before assembly
    Common in support corpora where the same policy appears in multiple docs.
  - retrieve_with_dedup() — convenience: retrieve + deduplicate in one call

In Week 3, this module gets replaced by LangGraph tool-based retrieval.
"""
import os
import json
from openai import OpenAI
from langfuse.decorators import observe, langfuse_context
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

TOP_K = 5


def get_connection():
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5434"),
        user=os.getenv("PG_USER", "workshop"),
        password=os.getenv("PG_PASSWORD", "workshop123"),
        dbname=os.getenv("PG_DATABASE", "acmera_kb"),
    )
    register_vector(conn)
    return conn


@observe(name="query_embedding")
def embed_query(query):
    response = client.embeddings.create(model="text-embedding-3-small", input=query)
    return response.data[0].embedding


@observe(name="retrieval")
def retrieve(query_embedding, top_k=TOP_K):
    """Standard dense retrieval — no filtering."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, doc_name, chunk_index, content, metadata,
                  1 - (embedding <=> %s::vector) AS similarity
           FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s""",
        (query_embedding, query_embedding, top_k),
    )
    results = []
    for row in cur.fetchall():
        results.append({
            "id": row[0], "doc_name": row[1], "chunk_index": row[2],
            "content": row[3],
            "metadata": row[4] if isinstance(row[4], dict) else json.loads(row[4]),
            "similarity": round(float(row[5]), 4),
        })
    cur.close()
    conn.close()

    langfuse_context.update_current_observation(metadata={
        "top_k": top_k, "filter": None,
        "results": [{"doc_name": r["doc_name"], "similarity": r["similarity"]} for r in results],
    })
    return results


@observe(name="retrieval_filtered")
def retrieve_filtered(query_embedding, doc_names: list[str], top_k=TOP_K):
    """
    Metadata-filtered dense retrieval (Session 3).
    Pre-filters to specific documents before semantic search.
    """
    conn = get_connection()
    cur = conn.cursor()

    placeholders = ",".join(["%s"] * len(doc_names))
    cur.execute(
        f"""SELECT id, doc_name, chunk_index, content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            WHERE doc_name IN ({placeholders})
            ORDER BY embedding <=> %s::vector
            LIMIT %s""",
        (query_embedding, *doc_names, query_embedding, top_k),
    )
    results = []
    for row in cur.fetchall():
        results.append({
            "id": row[0], "doc_name": row[1], "chunk_index": row[2],
            "content": row[3],
            "metadata": row[4] if isinstance(row[4], dict) else json.loads(row[4]),
            "similarity": round(float(row[5]), 4),
        })
    cur.close()
    conn.close()

    langfuse_context.update_current_observation(metadata={
        "top_k": top_k, "filter": doc_names,
        "results": [{"doc_name": r["doc_name"], "similarity": r["similarity"]} for r in results],
    })
    return results


def deduplicate_chunks(chunks: list, similarity_threshold: float = 0.75) -> list:
    """
    Remove near-duplicate chunks using word-level Jaccard similarity (Session 4).

    Support corpora often have the same policy text repeated across multiple
    documents (e.g., the 30-day return window appears in return_policy.md AND
    support_faq.md AND corporate_gifting.md). Deduplication prevents the LLM
    from seeing the same fact three times, wasting context budget.

    Args:
        chunks: List of retrieved chunk dicts
        similarity_threshold: Jaccard threshold — 0.75 means 75% word overlap

    Returns:
        Deduplicated list preserving original relevance order
    """
    seen_words = []
    unique = []

    for chunk in chunks:
        words = set(chunk["content"].lower().split())
        is_dup = False
        for seen in seen_words:
            if not words or not seen:
                continue
            intersection = len(words & seen)
            union = len(words | seen)
            if union > 0 and intersection / union >= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(chunk)
            seen_words.append(words)

    return unique


def retrieve_with_dedup(query_embedding, doc_names: list[str] | None = None,
                        top_k: int = TOP_K + 3) -> list:
    """
    Retrieve more candidates than needed, then deduplicate down to top_k.
    The extra candidates compensate for chunks that will be removed as duplicates.

    Args:
        query_embedding: Dense query embedding
        doc_names: Optional filter (None = search all docs)
        top_k: How many unique chunks to return after dedup

    Returns:
        Deduplicated list of chunks
    """
    candidates_needed = top_k + 3  # Fetch extra to absorb dedup losses
    if doc_names:
        candidates = retrieve_filtered(query_embedding, doc_names, top_k=candidates_needed)
    else:
        candidates = retrieve(query_embedding, top_k=candidates_needed)

    deduped = deduplicate_chunks(candidates)
    return deduped[:top_k]


@observe(name="context_assembly")
def assemble_context(retrieved_chunks):
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Source: {chunk['doc_name']}, Chunk {chunk['chunk_index']}]\n{chunk['content']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    langfuse_context.update_current_observation(metadata={
        "num_chunks": len(retrieved_chunks),
        "total_context_chars": len(context),
    })
    return context
