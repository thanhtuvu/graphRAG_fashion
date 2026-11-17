"""MCP server: exposes the retail knowledge-graph tools to Claude Desktop (stdio).

Claude Desktop launches this file as a child process (python.exe mcp_server.py)
and talks to it over stdin/stdout. Rule #1 for stdio servers: NEVER print() to
stdout - it is the wire protocol. Logs go to stderr.
"""
import logging
import os
import sys

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from retail_service import RetailService

load_dotenv()  # finds customer-graph/.env by searching upward from this file
logging.basicConfig(level=logging.INFO, stream=sys.stderr)

mcp = FastMCP("retail-graph")

_service: RetailService | None = None


def svc() -> RetailService:
    """Create the service on first tool call, so the server starts instantly
    (loading the embedding model takes a few seconds)."""
    global _service
    if _service is None:
        _service = RetailService(
            os.getenv("NEO4J_URI")
            , os.getenv("NEO4J_USERNAME")
            , os.getenv("NEO4J_PASSWORD")
        )
    return _service


@mcp.tool()
async def search_products(prompt_text: str) -> list[dict]:
    """Search product text based on the user prompt and return the most semantically
    similar products. Re-order or filter further based on additional context from the user."""
    return await svc().get_products_similar_text(prompt_text)


@mcp.tool()
async def recommend_products(segment_item_ids_or_codes: list[int]) -> list[dict]:
    """Retrieve product recommendations given a list of product codes, article ids, or
    segment ids. Re-order or filter further based on additional context from the user."""
    return await svc().get_product_recommendations(segment_item_ids_or_codes=segment_item_ids_or_codes)


@mcp.tool()
async def get_product_order_supplier_info(product_codes: list[int]) -> list[dict]:
    """Do NOT use without explicit product codes. Given a list of product codes, returns
    total orders and refunds, plus supplier info, for each product."""
    return await svc().get_product_order_supplier_info(product_codes=product_codes)


@mcp.tool()
async def get_supplier_order_product_info(supplier_ids: list[int]) -> list[dict]:
    """Do NOT use without explicit supplier ids. Given a list of supplier ids, returns
    total orders and refunds, plus product info, for each supplier."""
    return await svc().get_supplier_order_product_info(supplier_ids=supplier_ids)


@mcp.tool()
async def answer_general_question(user_question: str) -> str:
    """Answer a question about the retail graph (customers, orders, articles, credit
    notes/refunds, suppliers) by turning it into a Cypher query. Returns the Cypher
    that was run plus the result rows."""
    return await svc().text_to_cypher_query(user_question=user_question)


if __name__ == "__main__":
    mcp.run(transport="stdio")