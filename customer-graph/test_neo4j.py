import os
import sys
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase
import truststore
truststore.inject_into_ssl()  # must run before the Neo4j driver connects

load_dotenv()
uri = os.getenv("NEO4J_URI")
print("URI repr:", repr(uri))

log = logging.getLogger("neo4j")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))

driver = GraphDatabase.driver(
    uri
    , auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
driver.verify_connectivity()
print("connected")