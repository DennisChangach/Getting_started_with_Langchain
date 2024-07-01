# Getting Started with Langchain

### Main Lamgchain Modules:
- Langsmith - used for observability, testing, monitoring
- Langderve - Deployment
- LLMs - Can plug any type of LLM
- Langchain: Agents, Chains, Retrieval Strategies


RAG - Retrieval Augmented Generation - technique for augmenting LLM knowledge with additional data. 

LCEL - Langchain Expression Language
output parsers can be used to format the LLM responses

Chains refer to a sequence of calls whether to an LLM, a tool or a data processing step.

Tools are interfaces that an agent, chain or LLM can interact with the world.

Agents: The core idea of agents is to use a language model to choose a sequence of actions to take. In chains a sequence of actions is hardcoded in code. In agents, a language model is used as a reasoninng engine to determine which actions to take and in which order.



Reference:
Langchain docs: https://python.langchain.com/docs/use_cases/question_answering/
