import asyncio
import os
from dotenv import load_dotenv
from retail_service import RetailService

load_dotenv()
service = RetailService(
    os.getenv("NEO4J_URI")
    , os.getenv("NEO4J_USERNAME")
    , os.getenv("NEO4J_PASSWORD")
)

question = "sweaters for spring. Nothing too warm"
products = asyncio.run(service.get_products_similar_text(question))

# for p in products:
#     print(p.get("productCode"), "|", p.get("name"), "|", (p.get("description") or ""))
for p in products:
    print(f"{p} \n\n")