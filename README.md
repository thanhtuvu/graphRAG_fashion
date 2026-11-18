# GraphRAG Examples
This repository contains multiple GraphRAG examples. 
1. __[customer-graph](customer-graph):__ End-to-end worked example for building a GraphRAG agent to accelerate customer and retail analytics.
2. __[patterns-app](patterns-app):__ A streamlit app for introducing and teaching example GraphRAG patterns.

# Cypher queries
1. MATCH (n)
// WHERE any(label IN labels(n)) WHERE label IN ['Article', 'Order', 'Chunk']
RETURN labels(n) AS entity, count(n) AS total_nodes;


# Agent notes: who generates the Cypher?

There are 2 ways to get a Cypher query:

## Option A: the MCP server calls Claude to generate the Cypher
- Tool: `answer_general_question`, which calls `text_to_cypher_query` 
- `Text2CypherRetriever` sends the graph schema (`ontos/text-to-cypher.json`), the `custom_prompt` template and the user's question to Claude through an API call (`self._llm`). The MCP server gets the Cypher statement back from Claude's API response, runs it on Aura, and returns the Cypher and the rows to the chat model, which reads them and writes the answer to the user. On failure it retries up to 3 times, so one question can trigger up to 3 API calls.
- Billing: schema reading and Cypher generation are billed separately to my **API account**. The chat model still uses chat usage to decide to call the tool, read the returned Cypher and rows, and write the answer.

## Option B: the chat model writes the Cypher
- Tools: `get_graph_schema` and `run_read_cypher`. The MCP server makes no LLM call, so there is no cost on my API account.
- The chat model calls `get_graph_schema`, which returns the schema text as a tool result (input tokens). The chat model then writes the Cypher itself (output tokens). Then it calls `run_read_cypher` with the Cypher it wrote, the MCP server runs it on Aura (read-only) and returns the rows to the chat model, which reads them and writes the answer to the user.
- Billing: everything goes into my Claude plan's usage limit.

In both options an LLM generates the Cypher, and the MCP server is the one that connects to Aura and runs it.

# Tools notes:


`VectorRetriever.search()` embeds the user's prompt and runs a Cypher query that returns as many nodes as `top_k` specifies. 
Each returned row arrives as a Neo4j `Record` object  which may look like this:
```cypher
RETURN node { .*, `textEmbedding`: null } AS node
     , labels(node) AS nodeLabels
     , elementId(node) AS elementId
     , elementId(node) AS id
     , score
```
Each `Record` is processed by the `node_record_formatter` function which takes  1 `Record` at a time as a parameter. The function puts the node's properties into `content` and everything else into the `metatdata`. These are the 2 elements of  `RetrieverResultItem` object which the function returns

Once all nodes completely formatted, they form into a list of `RetrieverResultItem`. By default the query returns all of the node's properties (except `textEmbedding` is set to `None`). So originally a list of `RetrieverResultItem` looks like this (prompt sent: "lightweight summer dress")

```python
[
  RetrieverResultItem(
      content={'textEmbedding': None
            ,'productCode': 921906
            ,'name': 'Spring'
            ,'description': 'Calf-length dress in an airy viscose weave with a collar, concealed buttons at the top and long raglan sleeves with buttoned cuffs. Relaxed fit with a gathered seam at the hips and above the hem. Unlined.'
            ,'text': '##Product\nName: Spring\nType: Dress\nCategory: Garment Full body\nDescription: Calf-length dress in an airy viscose weave with a collar, concealed buttons at the top and long raglan sleeves with buttoned cuffs. Relaxed fit with a gathered seam at the hips and above the hem. Unlined.'
            ,'url': 'https://representative-domain/product/921906'
      }
    , metadata={'score': 0.71, 'nodeLabels': ['Product'], 'id': '4:abc...:12'}
  )
, RetrieverResultItem(content={...}, metadata={...})  # up to top_k items
]
```

But `return_properties` param controls which properties we will see which are `productCode` and `text` only. As a result, each `Record` now carries a smaller `node` map, so `content` holds only those two properties as below:

```python
[
  RetrieverResultItem(
      content={'productCode': 921906
            ,'text': '##Product\nName: Spring\nType: Dress\nCategory: Garment Full body\nDescription: Calf-length dress in an airy viscose weave with a collar, concealed buttons at the top and long raglan sleeves with buttoned cuffs. Relaxed fit with a gathered seam at the hips and above the hem. Unlined.'

      }
    , metadata={'score': 0.71, 'nodeLabels': ['Product'], 'id': '4:abc...:12'}
  )
, RetrieverResultItem(content={...}, metadata={...})  # up to top_k items
]
```

search() then wraps this list into a `RetrieverResult` which has 2 elements: `items` which is the list of `RetrieverResultItem` and `metadata` (retriever-level info, including the embedded prompt as `query_vector`)

`get_products_similar_text` only cares about the `items` element because it wants to access each item of the list and return the list of all product's nodes properties.
 