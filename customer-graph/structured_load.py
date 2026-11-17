"""Stage 2 (code route): load the 5 CSVs into Aura, same model as customer-struct-import.json.

MERGE on the model's key properties -> reruns are safe, and Stage 1's
Article / Order nodes get enriched instead of duplicated.
Run from the customer-graph folder:  python structured_load.py
"""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI")
    , auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

DATA = Path("data")
BATCH = 2_000

# (label, key property) - mirrors the uniqueness constraints in the model
CONSTRAINTS = [
    ("Article", "articleId")
    , ("Order", "orderId")
    , ("Customer", "customerId")
    , ("Supplier", "supplierId")
    , ("Product", "productCode")
    , ("ProductType", "name")
    , ("ProductCategory", "name")
]


def records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> plain dicts, NaN/NA -> None (so Neo4j stores null, not NaN)."""
    return [
        {k: (None if pd.isna(v) else v) for k, v in r.items()}
        for r in df.to_dict("records")
    ]


def load(session, title: str, query: str, df: pd.DataFrame) -> None:
    rows = records(df)
    for i in range(0, len(rows), BATCH):
        session.run(query, rows=rows[i:i + BATCH]).consume()
    print(f"{title}: {len(rows):,} rows")


# ---- read CSVs (IDs that are hashes/strings stay strings) -------------------
suppliers = pd.read_csv(DATA / "suppliers.csv")
products = pd.read_csv(DATA / "products.csv")
articles = pd.read_csv(DATA / "articles.csv")
customers = pd.read_csv(DATA / "customers.csv", dtype={"customerId": str, "postalCode": str})
details = pd.read_csv(DATA / "order-details.csv", dtype={"customerId": str})

orders = details.drop_duplicates("orderId")[["orderId", "tDat", "customerId"]]
# an article can repeat inside one order; MERGE collapses those to one CONTAINS edge
lines = details.drop_duplicates(["orderId", "articleId"])[["orderId", "articleId", "price"]]

with driver.session() as session:
    for label, prop in CONSTRAINTS:
        session.run(
            f"CREATE CONSTRAINT {prop}_{label}_uniq IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
        ).consume()
    print("constraints ready")

    load(session, "Supplier", """
        UNWIND $rows AS r
        MERGE (s:Supplier {supplierId: r.supplierId})
        SET s.name = r.supplierName
          , s.address = r.supplierAddress
    """, suppliers)

    load(session, "Product + type/category", """
        UNWIND $rows AS r
        MERGE (p:Product {productCode: r.productCode})
        SET p.name = r.prodName
          , p.description = r.detailDesc
        MERGE (t:ProductType {name: r.productTypeName})
        MERGE (c:ProductCategory {name: r.productGroupName})
        MERGE (p)-[:PART_OF]->(t)
        MERGE (p)-[:PART_OF]->(c)
    """, products)

    load(session, "Article + VARIANT_OF/SUPPLIED_BY", """
        UNWIND $rows AS r
        MERGE (a:Article {articleId: r.articleId})
        SET a.colourGroupCode = r.colourGroupCode
          , a.colourGroupName = r.colourGroupName
          , a.graphicalAppearanceNo = r.graphicalAppearanceNo
          , a.graphicalAppearanceName = r.graphicalAppearanceName
        WITH a, r
        MATCH (p:Product {productCode: r.productCode})
        MATCH (s:Supplier {supplierId: r.supplierId})
        MERGE (a)-[:VARIANT_OF]->(p)
        MERGE (a)-[:SUPPLIED_BY]->(s)
    """, articles)

    load(session, "Customer", """
        UNWIND $rows AS r
        MERGE (c:Customer {customerId: r.customerId})
        SET c.age = toInteger(r.age)
          , c.clubMemberStatus = r.clubMemberStatus
          , c.fashionNewsFrequency = r.fashionNewsFrequency
          , c.postalCode = r.postalCode
    """, customers)

    load(session, "Order + ORDERED", """
        UNWIND $rows AS r
        MERGE (o:Order {orderId: r.orderId})
        SET o.date = datetime(r.tDat)
        WITH o, r
        MATCH (c:Customer {customerId: r.customerId})
        MERGE (c)-[:ORDERED]->(o)
    """, orders)

    load(session, "CONTAINS", """
        UNWIND $rows AS r
        MATCH (o:Order {orderId: r.orderId})
        MATCH (a:Article {articleId: r.articleId})
        MERGE (o)-[c:CONTAINS]->(a)
        SET c.price = r.price
    """, lines)

driver.close()
print("done")