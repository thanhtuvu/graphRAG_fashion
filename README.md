# GraphRAG Examples
This repository contains multiple GraphRAG examples. 
1. __[customer-graph](customer-graph):__ End-to-end worked example for building a GraphRAG agent to accelerate customer and retail analytics.
2. __[patterns-app](patterns-app):__ A streamlit app for introducing and teaching example GraphRAG patterns.

# Cypher queries
1. MATCH (n)
// WHERE any(label IN labels(n)) WHERE label IN ['Article', 'Order', 'Chunk']
RETURN labels(n) AS entity, count(n) AS total_nodes;

# Study notes:


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
 