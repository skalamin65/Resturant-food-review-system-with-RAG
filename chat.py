from embedding_maneger import EmbeddingManeger
from vector_store import VectorStore
from rag_retriver import RAGRetriver
from langchain_ollama import ChatOllama


# Load embedding manager
embedding_manager = EmbeddingManeger()

# Connect to the existing persistent vector store
vector_store = VectorStore(
    persist_directory="data/vector_store",
    collection_name="restaurant_documents"
)

# Initialize retriever
rag_retriever = RAGRetriver(
    embedding_manager,
    vector_store
)

# Initialize LLM
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.4,
    num_predict=1024
)


def generate_output(query, rag_retriever, llm, top_k=3):
    results = rag_retriever.retrive(query, top_k)

    context = "\n".join(doc["document"] for doc in results) if results else ""

    if not context:
        return "I couldn't find relevant information in the restaurant data."

    prompt = f"""
    You are a restaurant assistant.

    Answer the question using only the supplied context.
    If the information is missing, say that you do not
    have that information.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = llm.invoke(prompt)
    return response.content


def main():
    print("Restaurant RAG Assistant is ready!")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Enter your query: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        answer = generate_output(
            question,
            rag_retriever,
            llm
        )

        print("\nAnswer:", answer, "\n")


if __name__ == "__main__":
    main()