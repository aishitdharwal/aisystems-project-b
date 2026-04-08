"""
Self-contained retrieval layer for Project B — Session 3 update.

New in Session 3:
  - retrieve_filtered()   — pre-filter by doc_name before semantic search
    Narrows the search space to the most relevant policy documents.
    This is metadata filtering — a pre-retrieval routing technique.

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
    Metadata-filtered dense retrieval.

    Pre-filters to specific documents before semantic search.
    This is pre-retrieval routing — same query, much smaller search space.

    Why it helps: searching all 19 docs for "return window" might surface
    promotional events or corporate gifting policy alongside the primary
    return policy. Filtering to the 2-3 most relevant docs improves precision.

    Args:
        query_embedding: Dense query embedding
        doc_names: List of doc filenames to search within
        top_k: Number of results to return

    Returns:
        List of chunk dicts matching the filter
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
