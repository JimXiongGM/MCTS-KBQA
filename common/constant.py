"""
We use flask to build the API server for the local LLMs. The following is the port mapping.
"""

# Model API server.
ACTOR_SERVERS = [
    # CUDA_VISIBLE_DEVICES=0 vllm serve save-model-sft/small-KQAPro-v2/Meta-Llama-3.1-8B-Instruct-epoch5 --enable-prefix-caching --api-key kbqa --port 28000 --gpu-memory-utilization 0.45
    "http://127.0.0.1:28000",
    # "http://127.0.0.1:28001",
]
COT_REWARD_SERVERS = [
    # local-vllm
    # CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --enable-prefix-caching --api-key kbqa --port 28500 --gpu-memory-utilization 0.45
    # "http://127.0.0.1:28500",
    # "http://127.0.0.1:28501",
    # "http://127.0.0.1:29000", # gemma
    # "http://127.0.0.1:29001", # qwen
]
LOGIC_FORM_REWARD_SERVERS = [
    # local-vllm
    # CUDA_VISIBLE_DEVICES=0 vllm serve save-model-sft/small-KQAPro-LogicForm/Meta-Llama-3.1-8B-Instruct-epoch5 --enable-prefix-caching --api-key kbqa --port 28600 --gpu-memory-utilization 0.45
    # "http://127.0.0.1:28600",
    "http://127.0.0.1:28601",
]

ACTOR_KEY = "kbqa"
REWARD_KEY = "kbqa"

"""
Short tool descriptions for fine-tuned WebQSP, CWQ, and KQA Pro models.
"""

# cwq/webqsp (for FB)
TOOL_DESC_SHORT_FB = """Given a question, you need to use the specific tools to interact with Freebase and write a SPARQL query to get the answer.

The following document includes the description of tools, the types of nodes and edges (KG schema), and some critical graph patterns.

1. SearchNodes(query)
Description: Search for nodes in the knowledge graph based on the surface name.\x20

2. ExecuteSPARQL(sparql)
Description: Execute a SPARQL query. You can explore the KG freely using this tool.

3. SearchGraphPatterns(sparql, semantic)
Description: Parameter `sparql` MUST start with "SELECT ?e WHERE". The tool will query the subgraphs with ?e as the head or tail entities, respectively, and return them together. The `semantic` parameter indicates the expected predicate semantics. If provided, the tool will sort the queried subgraphs by semantics. If not, the tool returns the entire subgraph. Note! If a predicate corresponds to multiple tail entities, this tool randomly returns one of them. It is worth noting that, in Freebase, due to the use of "Compound Value Type" (CVT) to represent an event, a one-hop relationship semantically requires two hops in Freebase. If encountering a CVT node, this tool will split the two-hop relationships involving the CVT node into predicate pairs and consider them as one-hop relationships.

Now, think and solve the following complex question step by step."""

# kqapro
TOOL_DESC_SHORT_KQAPRO = """Given a question, you need to use the specific tools to interact with Wikidata (KQA Pro version) and write a SPARQL query to get the answer.

The following document includes the description of tools, the types of nodes and edges (KG schema), and some critical graph patterns.

1. SearchNodes(query)
Description: Search for nodes in the knowledge graph based on the surface name. There are two types of nodes: entity and concept.

2. ExecuteSPARQL(sparql)
Description: Execute a SPARQL query. You can explore the KG freely using this tool.

3. SearchGraphPatterns(sparql, semantic)
Description: The parameter sparql MUST start with "SELECT ?e WHERE". The tool will query the subgraphs with ?e as the head or tail entities, respectively, and return them together. The semantic parameter indicates the expected predicate semantics. If provided, the tool will sort the queried subgraphs by semantics. If not, the tool returns the entire subgraph. Note! This tool will return a randomly instantiated triple; you should pay attention to the semantics of the predicate, not the specific names of the head entity or the tail entity.

Now, think and solve the following complex question step by step."""
