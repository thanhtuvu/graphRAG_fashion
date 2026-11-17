import asyncio, os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.experimental.components.pdf_loader import DataLoader
from neo4j_graphrag.experimental.components.text_splitters.langchain import LangChainTextSplitterAdapter
from neo4j_graphrag.experimental.components.types import PdfDocument, DocumentInfo
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import AnthropicLLM
from neo4j_graphrag.llm.types import LLMResponse
from rag_schema_from_onto import getSchemaFromOnto

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

neo4j_schema = getSchemaFromOnto("ontos/customer.ttl")
print(neo4j_schema)


# Create DocumentLoader
class PdfLoaderWithPageBreaks(DataLoader):
    async def run(self, filepath: Path) -> PdfDocument:
        loader = PyPDFLoader(filepath)
        text = ''
        async for page in loader.alazy_load():
            text = text + " __PAGE__BREAK__ " + page.page_content
        return PdfDocument(
            text=text,
            document_info=DocumentInfo(path=filepath), )


# Create a Splitter object
splitter = LangChainTextSplitterAdapter(
    CharacterTextSplitter(chunk_size=15_000, chunk_overlap=0, separator=" __PAGE__BREAK__ ")
)

# Create an Embedder object (local model, no API key; downloads on first run)
embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")

# LLM wrapper that tolerates thinking blocks (see class docstring)
class TextOnlyAnthropicLLM(AnthropicLLM):
    """AnthropicLLM that skips thinking blocks instead of assuming content[0] is text.

    neo4j-graphrag 1.7.0 reads response.content[0].text. Newer Claude models can put
    a thinking block first, which crashes with "'ThinkingBlock' has no attribute 'text'".
    """

    @staticmethod
    def _first_text(response) -> LLMResponse:
        for block in response.content:
            if block.type == "text":
                return LLMResponse(content=block.text)
        raise LLMGenerationError("LLM returned no text block.")

    def invoke(self, input, message_history=None, system_instruction=None):
        try:
            response = self.client.messages.create(
                model=self.model_name
                , system=system_instruction or self.anthropic.NOT_GIVEN
                , messages=self.get_messages(input, message_history)
                , **self.model_params
            )
            return self._first_text(response)
        except self.anthropic.APIError as e:
            raise LLMGenerationError(e)

    async def ainvoke(self, input, message_history=None, system_instruction=None):
        try:
            response = await self.async_client.messages.create(
                model=self.model_name
                , system=system_instruction or self.anthropic.NOT_GIVEN
                , messages=self.get_messages(input, message_history)
                , **self.model_params
            )
            return self._first_text(response)
        except self.anthropic.APIError as e:
            raise LLMGenerationError(e)


# Instantiate the LLM (reads ANTHROPIC_API_KEY from the environment)
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
model_params = {"max_tokens": 8192}  # required by the Anthropic API

if CLAUDE_MODEL.startswith("claude-sonnet-5-5"):
    # Sonnet 5.5 thinks by default; "between_tools" is its lowest thinking setting
    model_params["thinking"] = {"type": "between_tools"}

llm = TextOnlyAnthropicLLM(
    model_name=CLAUDE_MODEL
    , model_params=model_params
)

# instantiate the SimpleKGPipeline
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    pdf_loader=PdfLoaderWithPageBreaks(),
    text_splitter=splitter,
    embedder=embedder,
    entities=list(neo4j_schema.entities.values()),
    relations=list(neo4j_schema.relations.values()),
    potential_schema=neo4j_schema.potential_schema,
    on_error="IGNORE",
    from_pdf=True,
)

# load credit notes
asyncio.run(kg_builder.run_async(file_path='data/credit-notes.pdf'))

# perform entity resolution
print("Performing Additional Entity Resolution")
driver.execute_query('''
MATCH (n:Article)
WITH n.articleId AS id, collect(n) as nodes
CALL apoc.refactor.mergeNodes(nodes, {
  properties: {
      `.*`: 'combine'
  },
  mergeRels: true
})
YIELD node
RETURN node;
''')

driver.execute_query('''
MATCH (n:Order)
WITH n.orderId AS id, collect(n) as nodes
CALL apoc.refactor.mergeNodes(nodes, {
  properties: {
      `.*`: 'combine'
  },
  mergeRels: true
})
YIELD node
RETURN node
''')

print("Removing Unneeded Nodes")
driver.execute_query('MATCH (n:Product) WHERE n:__Entity__ DETACH DELETE n')

driver.close()