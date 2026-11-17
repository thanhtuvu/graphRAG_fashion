"""Stage 3: Product text + local embeddings + vector index (no OpenAI).

Run from the customer-graph folder:  python ingest_post_processing.py
Safe to rerun: SET overwrites, the index uses IF NOT EXISTS.
"""
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI")
    , auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# Same model as Stage 1's embedder. The agent (Stage 4) MUST embed queries with it too.
EMBED_MODEL = "all-MiniLM-L6-v2"
DIMENSION = 384  # all-MiniLM-L6-v2 output size (OpenAI ada-002 was 1536)
INDEX_NAME = "product_text_embeddings"
BATCH = 256

# 1) text + url properties for every product
print("Formatting Product Text")
res = driver.execute_query('''
MATCH (p:Product)
OPTIONAL MATCH (p)-[:PART_OF]->(c:ProductCategory)
OPTIONAL MATCH (p)-[:PART_OF]->(t:ProductType)
SET p.text = '##Product\\n' +
    'Name: ' + coalesce(p.name, '') + '\\n' +
    'Type: ' + coalesce(t.name, '') + '\\n' +
    'Category: ' + coalesce(c.name, '') + '\\n' +
    'Description: ' + coalesce(p.description, ''),
    p.url = 'https://representative-domain/product/' + toString(p.productCode)
RETURN count(p) AS propertySetCount
''')
print(f"  text set on {res.records[0]['propertySetCount']:,} products")

# 2) embed locally, in batches
print(f"Creating Product Text Embeddings ({EMBED_MODEL})")
model = SentenceTransformer(EMBED_MODEL)
records, _, _ = driver.execute_query('''
MATCH (p:Product)
WHERE p.description IS NOT NULL AND p.description <> ''
RETURN p.productCode AS productCode, p.text AS text
''')
rows = [r.data() for r in records]

with driver.session() as session:  # default (home) database; Aura's isn't always called "neo4j"
    for i in tqdm(range(0, len(rows), BATCH)):
        batch = rows[i:i + BATCH]
        vectors = model.encode([r["text"] for r in batch])
        session.run('''
        UNWIND $rows AS r
        MATCH (p:Product {productCode: r.productCode})
        CALL db.create.setNodeVectorProperty(p, 'textEmbedding', r.vector)
        ''', rows=[
            {"productCode": r["productCode"], "vector": v.tolist()}
            for r, v in zip(batch, vectors)
        ]).consume()
print(f"  embedded {len(rows):,} products")

# 3) vector index (dimension must match the model!)
print("Creating Product Vector Index")
driver.execute_query('''
CREATE VECTOR INDEX product_text_embeddings IF NOT EXISTS FOR (n:Product) ON (n.textEmbedding)
OPTIONS {indexConfig: {
 `vector.dimensions`: toInteger($dimension),
 `vector.similarity_function`: 'cosine'
}}
''', dimension=DIMENSION)

# wait for index to come online
driver.execute_query('CALL db.awaitIndex("product_text_embeddings", 300)')

# 4) smoke test: does semantic search return sensible products?
query = "lightweight summer dress"
hits = driver.execute_query('''
CALL db.index.vector.queryNodes($index, 3, $vec) YIELD node, score
RETURN node.name AS name, score
''', index=INDEX_NAME, vec=model.encode(query).tolist())
print(f"Smoke test: '{query}'")
for h in hits.records:
    print(f"  {h['score']:.3f}  {h['name']}")

driver.close()