from langchain_core.vectorstores import VectorStoreRetriever
from dotenv import load_dotenv

load_dotenv()

CHUNK_SEPARATOR = "\n\n---CHUNK---\n\n"

def build_retrieve_tool(retriever: VectorStoreRetriever):

    def retrieve(query: str) -> str:
        """Retrieve relevant information from the vector store based on the query."""
        docs = retriever.invoke(query)
        result = CHUNK_SEPARATOR.join([doc.page_content.strip() for doc in docs])
        return result

    return retrieve