import asyncio
import logging
import re
from pathlib import Path

from neo4j import GraphDatabase, Query, READ_ACCESS, RoutingControl
from typing import List
from customer_schema import Product, CustomerSegment, Supplier, ProductInfo, SupplierInfo
from neo4j_graphrag.retrievers import VectorCypherRetriever, Text2CypherRetriever, VectorRetriever
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from formatters import node_record_formatter
from claude_llm import build_claude_llm


MAX_ROWS = 50  # cap on rows returned to the chat model
QUERY_TIMEOUT_S = 15  # server-side transaction timeout
_BLOCKED_CYPHER = re.compile(r"\bload\s+csv\b|\bapoc\.", re.IGNORECASE)  # network access / extension procedures

def _json_safe(value):
    """Make a Neo4j record value JSON-safe and drop embedding vectors
    (384 floats per product would flood the chat)."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items() if k != "textEmbedding"}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

EXAMPLES=[
    "USER INPUT: 'How many orders are there?' QUERY: MATCH (o:Order) RETURN count(o) AS orders"
    , "USER INPUT: 'What is the total refund amount?' QUERY: MATCH (c:CreditNote) RETURN sum(c.amount) AS totalRefund"
    , "USER INPUT: 'Which 5 suppliers have the most refunded articles?' QUERY: MATCH (:CreditNote)-[:REFUND_OF_ARTICLE]->(:Article)-[:SUPPLIED_BY]->(s:Supplier) RETURN s.name AS supplier, count(*) AS refunds ORDER BY refunds DESC LIMIT 5"
],

class RetailService:
    def __init__(self, uri, user, pwd):
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        self._driver = driver
        self._embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")
        self._llm = build_claude_llm()

    def warm_up(self) -> None:
        """Pay the cold-start costs now: the first embedding (model inference) and the Aura connection."""
        self._embedder.embed_query("warm up")
        self._driver.verify_connectivity()

    async def get_products_similar_text(self, prompt_text: str) -> List[Product]:
        retriever = VectorRetriever(
            driver=self._driver,
            index_name="product_text_embeddings",
            embedder=self._embedder,
            result_formatter=node_record_formatter,
            return_properties=["productCode", "text"]
        )

        retriever_result = retriever.search(query_text=prompt_text, top_k=5)

        products = []
        for item in retriever_result.items:
            p = item.content
            products.append(p)

        return products

    async def get_product_recommendations(self, segment_item_ids_or_codes: List[int]) -> List[Product]:
        res = self._driver.execute_query("""
        MATCH (customer:Customer)-[:ORDERED]->()-[:CONTAINS]->()-[:VARIANT_OF]->
        (interestedInProducts:Product)<-[:VARIANT_OF]-(interestedInArticles:Article)<-[:CONTAINS]-()<-[:ORDERED]
        -(:Customer)-[:ORDERED]->()-[:CONTAINS]->(recArticle:Article)-[:VARIANT_OF]->(product:Product)
        WHERE (interestedInArticles.articleId IN $itemIds) 
            OR (interestedInProducts.productCode IN $itemIds)
        WITH count(recArticle) AS recommendationScore, product
        RETURN product ORDER BY recommendationScore DESC LIMIT 20
        """, itemIds=segment_item_ids_or_codes)

        products = []
        for item in res.records:
            src = item.data()['product']
            s: Product = {k: src[k] for k in Product.__annotations__ if k in src}
            products.append(s)
        return products


    async def get_supplier_order_product_info(self, supplier_ids: List[int]) -> list[SupplierInfo]:
        res = self._driver.execute_query("""
        MATCH(p:Product)<-[:VARIANT_OF]-(:Article)-[:SUPPLIED_BY]->(s)
        WHERE s.supplierId IN $supplierIds
        WITH DISTINCT p, s,
          COUNT {MATCH (:Order)-[:CONTAINS]->(a)-[:VARIANT_OF]->(p) 
                WHERE (a)-[:SUPPLIED_BY]->(s)
                } AS numberOfOrders, 
          COUNT {MATCH (:CreditNote)-[:REFUND_OF_ARTICLE]-(a)-[:VARIANT_OF]->(p) 
                WHERE (a)-[:SUPPLIED_BY]->(s)
                } AS numberOfRefunds
        RETURN s.supplierId AS supplierId,  
            s.name as supplier_name,
          sum(numberOfOrders) AS totalOrders, 
          sum(numberOfRefunds) AS totalReturns,
          collect({productCode:p.productCode, product_name:p.name, numberOfOrders:numberOfOrders, numberOfRefunds:numberOfRefunds}) AS supplierInfos
        """, supplierIds=supplier_ids)

        supplier_infos = []
        for item in res.records:
            supplier_info: SupplierInfo = item.data()
            supplier_infos.append(supplier_info)
        return supplier_infos

    # ---- Option A: the server calls the LLM to generate the Cypher (then runs it).
    async def text_to_cypher_query(self, user_question: str) -> str:
        schema_path = Path(__file__).resolve().parent.parent / "ontos" / "text-to-cypher.json"
        with open(schema_path, "r", encoding="utf-8") as file:
            query_schema = file.read()

        retriever = Text2CypherRetriever(
            driver=self._driver,
            llm=self._llm,
            neo4j_schema=query_schema,
            custom_prompt="""
Task: Generate a Cypher statement for querying a Neo4j graph database from a user input. 
-Do not include triple backticks ``` or ```cypher or any additional text except the generated Cypher statement in your response.
-Do not use any properties or relationships not included in the schema.

Schema:
{schema}

Examples:
{EXAMPLES}

Input:
{query_text}
"""
        )

        max_retries = 3
        attempt = 0
        errors = []

        while attempt < max_retries:
            try:
                retriever_result = retriever.search(query_text=user_question)

                answer = ""
                cypher = retriever_result.metadata['cypher']
                logging.info(f"Text2Cypher Query:\n{cypher}")
                for item in retriever_result.items:
                    content = str(item.content)
                    if content:
                        answer += content + '\n\n'

                return f"Cypher used:\n{cypher}\n\nResults:\n{answer}"

            except Exception as e:
                error_message = f"\nError on last attempt number {attempt + 1}: {str(e)}"
                errors.append(error_message)
                user_question += error_message  # Append errors to user question for LLM awareness
                attempt += 1

                if attempt < max_retries:
                    await asyncio.sleep(1)  # Small delay before retrying

        # If all retries fail, return an error message
        return f"Failed after {max_retries} attempts. Errors: {errors}"

    # ---- Option B: the chat model writes the Cypher
    async def run_read_cypher(self, query: str) -> dict:
        """Run a read-only Cypher query that the chat model wrote."""
        if _BLOCKED_CYPHER.search(query):
            raise ValueError("LOAD CSV and apoc.* procedures are not allowed.")

        # Step 1: EXPLAIN plans the query without running it, and the server reports its
        # type: 'r' = read-only, 'rw'/'w' = writes, 's' = schema change.
        plan = self._driver.execute_query("EXPLAIN " + query, routing_=RoutingControl.READ)
        if plan.summary.query_type != "r":
            raise ValueError(
                f"Only read-only queries are allowed (the server classified this one as '{plan.summary.query_type}')."
            )

        # Step 2: run in a read-access session with a timeout; fetch one extra row to detect truncation.
        logging.info(f"run_read_cypher query:\n{query}")
        with self._driver.session(default_access_mode=READ_ACCESS) as session:
            result = session.run(Query(query, timeout=QUERY_TIMEOUT_S))
            records = result.fetch(MAX_ROWS + 1)

        rows = [_json_safe(r.data()) for r in records[:MAX_ROWS]]
        return {"row_count": len(rows), "truncated": len(records) > MAX_ROWS, "rows": rows}