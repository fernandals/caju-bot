import json
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


if __name__ == "__main__":

    discipline_name = "Software Architecture"
    base_path = Path("data/knowledge_bases")

    persist_dir = None
    embedding_model = os.getenv(
        "EMBED_MODEL",
        "text-embedding-3-large"
    )

    # procura a KB correta
    for kb_dir in base_path.iterdir():

        metadata_path = kb_dir / "metadata.json"

        if not metadata_path.exists():
            continue

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if metadata.get("name") == discipline_name:

            persist_dir = kb_dir / "chroma"

            embedding_model = metadata.get(
                "embedding_model",
                embedding_model
            )

            break

    if persist_dir is None:
        raise ValueError(
            f"Knowledge base '{discipline_name}' não encontrada."
        )

    if not persist_dir.exists():
        raise ValueError(
            f"Diretório Chroma não encontrado: {persist_dir}"
        )

    print(f"Usando KB: {discipline_name}")
    print(f"Persist dir: {persist_dir}")
    print(f"Embedding model: {embedding_model}")

    # embeddings
    embeddings = OpenAIEmbeddings(
        model=embedding_model
    )

    # carrega vectorstore
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings
    )

    # debug bruto do banco
    raw_data = vectorstore.get()

    print("\n" + "=" * 60)
    print("DEBUG VECTORSTORE")
    print("=" * 60)

    print(f"Quantidade de documentos: {len(raw_data['documents'])}")

    if raw_data["documents"]:
        print("\nPrimeiro documento:")
        print(raw_data["documents"][0][:300])

    if raw_data["metadatas"]:
        print("\nPrimeiro metadata:")
        print(json.dumps(raw_data["metadatas"][0], indent=2))

    # retriever
    retriever = vectorstore.as_retriever(
      search_type="mmr",
      search_kwargs={
          "k": 5,
          "fetch_k": 20,
          "lambda_mult": 0.7
      }
    )

    query = "What is software architecture?"

    print("\n" + "=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    docs = retriever.invoke(query)

    if not docs:
        print("Nenhum documento encontrado.")
        exit()

    for i, doc in enumerate(docs, 1):

        print(f"\nResultado {i}")
        print("-" * 50)

        print("Conteúdo (300 chars):")
        print(doc.page_content[:300])

        print("\nMetadata:")

        if not doc.metadata:
            print("Metadata vazio.")
        else:
            for key, value in doc.metadata.items():
                print(f"{key}: {value}")