"""MCP server: exposes the retail knowledge-graph tools to Claude Desktop (stdio).

Claude Desktop launches this file as a child process (python.exe mcp_server.py)
and talks to it over stdin/stdout. Rule #1 for stdio servers: NEVER print() to
stdout - it is the wire protocol. Logs go to stderr.
"""
import logging
import os
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from retail_service import RetailService, EXAMPLES

load_dotenv()  # finds customer-graph/.env by searching upward from this file
logging.basicConfig(level=logging.INFO, stream=sys.stderr)

mcp = FastMCP("retail-graph")

_service: RetailService | None = None
_service_lock = threading.Lock()


def svc() -> RetailService:
    """Return the shared RetailService, building it on first use (loading the embedding model
    takes a while). The lock keeps a tool call and the background warm-up from building it twice."""
    global _service
    with _service_lock:
        if _service is None:
            _service = RetailService(
                os.getenv("NEO4J_URI")
                , os.getenv("NEO4J_USERNAME")
                , os.getenv("NEO4J_PASSWORD")
            )
    return _service


def _warm_up() -> None:
    """Runs in a background thread right after launch, so the cold-start cost is paid before the
    first tool call. A failure here is only logged; tool calls surface their own errors."""
    try:
        svc().warm_up()
        logging.info("warm-up done")
    except Exception:
        logging.exception("warm-up failed")


@mcp.tool()
async def search_products(prompt_text: str) -> list[dict]:
    """Search product text based on the user prompt and return the most semantically
    similar products. Re-order or filter further based on additional context from the user."""
    return await svc().get_products_similar_text(prompt_text)


@mcp.tool()
async def recommend_products(segment_item_ids_or_codes: list[int]) -> list[dict]:
    """Retrieve product recommendations given a list of product codes, article ids, or
    segment ids. Re-order or filter further based on additional context from the user."""
    return await svc().get_product_recommendations(segment_item_ids_or_codes)


@mcp.tool()
async def get_supplier_order_product_info(supplier_ids: list[int]) -> list[dict]:
    """Do NOT use without explicit supplier ids. Given a list of supplier ids, returns
    total orders and refunds, plus product info, for each supplier."""
    return await svc().get_supplier_order_product_info(supplier_ids=supplier_ids)


# ---- Option A: the server calls the LLM to generate the Cypher.
@mcp.tool()
async def answer_general_question(user_question: str) -> str:
    """Answer a question about the retail graph (customers, orders, articles, credit
    notes/refunds, suppliers) by turning it into a Cypher query. Returns the Cypher
    that was run plus the result rows."""
    return await svc().text_to_cypher_query(user_question=user_question)


# ---- Option B: the chat model writes the Cypher.
@mcp.tool()
async def get_graph_schema() -> str:
    """Return the graph schema (node labels, relationship types, properties) as JSON,
    followed by example questions with their Cypher. Call this BEFORE writing any Cypher for run_read_cypher."""
    schema = (Path(__file__).resolve().parent.parent / "ontos" / "text-to-cypher.json").read_text(encoding="utf-8")
    return schema + "\n\nExamples:\n" + "\n".join(EXAMPLES)


@mcp.tool()
async def run_read_cypher(query: str) -> dict:
    """Run a read-only Cypher query on the retail graph and return up to 50 rows.
    Rules: call get_graph_schema first; do not use any labels, relationship types or properties
    that are not in the schema; never return embedding properties (textEmbedding); write queries
    are rejected. If the query fails, fix it using the error message and try again."""
    return await svc().run_read_cypher(query)


if __name__ == "__main__":
    threading.Thread(target=_warm_up, daemon=True).start()
    mcp.run(transport="stdio")