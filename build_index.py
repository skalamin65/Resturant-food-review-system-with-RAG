import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embedding_maneger import EmbeddingManeger
from vector_store import VectorStore


def load_all_text():
    folder_path = "Data/txt"
    all_docs = []

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"Folder not found: {folder_path}"
        )

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            loader = TextLoader(
                file_path,
                encoding="utf-8"
            )
            doc=loader.load()
            all_docs.extend(doc)

    print("Total documents loaded:", len(all_docs))
    return all_docs

# Converting documents into chunks
def split_documents(documents, chunk_size=200, chunk_overlap=30):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)
    print("Total chunks created:", len(chunks))

    return chunks


def main():
    # Use the same persistent directory and collection name
    # in both build_index.py and chat.py.
    vector_store = VectorStore(
        persist_directory="data/vector_store",
        collection_name="restaurant_documents"
    )

    # Avoid indexing the same documents again.
    if vector_store.collection.count() > 0:
        print("Existing index found. Skipping indexing.")
        print(
            "Stored chunks:",
            vector_store.collection.count()
        )
        return

    documents = load_all_text()
    chunks = split_documents(documents)

    embedding_manager = EmbeddingManeger()

    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)

    vector_store.add_document(chunks, embeddings)

    print("Index created and saved successfully!")


if __name__ == "__main__":
    main()