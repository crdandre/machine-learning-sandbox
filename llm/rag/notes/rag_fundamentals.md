# Notes on Retrieval Augmented Generation (RAG)

## What is RAG?

RAG is a context-setting technique for prompting LLMs to answer specific questions about specific, user-provided data. It involves an intermediate step which queries a vector database for embeddings which corrsepond to a prompt, then prompting an LLM with this info to answer a question with specific information.

Example: A user wants to ask a textbook a question. The textbook contents are fed into a text-embedding model which creates a text embedding (i.e. a real vector representation for each sentence or arbitrary unit of text). The prompt to the LLM is processed similarly. The raw text of both the LLM prompt and context found in the vector database are passed into the LLM which can respond more accurately than by only using the original prompt, given the relevant context.

## RAG Strategies
### Simple
That described above. One prompt, one vector database, one retrieved context within the model's input token limit, one response. This can be useful for asking standalone questions for fact retrieval.

### Next-Level Concepts within RAG

### Other RAG-Adjacent-and-Combinable Techniques

#### Agents

## RAG Tools
### Prompting Frameworks
- [LangChain](https://www.langchain.com/)
- [ell](https://docs.ell.so/index.html)

### Orchestration
- [n8n](https://n8n.io/)
- [Zapier](https://zapier.com/)
- Local solutions? Is n8n local-able?

### Vector Databases
- [LanceDB](https://lancedb.github.io/lancedb/)
- [Pinecone](https://www.pinecone.io/)
- [Supabase](https://supabase.com/)
- Is there a containerized equivalent for the supabase flexible setup from n8n-supabase-rag?

## References
1. [Overview of RAG (Basics, Agents, etc.)](https://www.youtube.com/watch?v=Y08Nn23o_mY)
2. [Series of Articles on RAG](https://medium.com/@j13mehul/rag-part-1-from-naive-to-advanced-cb40674a7738)
3. [RAG Agent w/ n8n and Supabase](https://www.youtube.com/watch?v=PEI_ePNNfJQ)