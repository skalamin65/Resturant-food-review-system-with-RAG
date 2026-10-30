import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedding_maneger import EmbeddingManeger
from vector_store import VectorStore
from rag_retriver import RAGRetriver
from langchain_ollama import ChatOllama
def load_all_text():
    folder_path="Data/txt"
    all_docs=[]
    num_docs=0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".txt"):
            file_path=os.path.join(folder_path,filename)
            loader=TextLoader(file_path)
            doc=loader.load()
            all_docs.extend(doc)
            num_docs+=1
    return all_docs
    print("Total txt file :",num_docs)
    print("num of docs: ",len(all_docs))

all_text_doc=load_all_text()
print(type(all_text_doc))

# convert documents into chunks
def split_documents(documents,chunks_size=200,chunk_overlap=30):
    text_spliter=RecursiveCharacterTextSplitter(
        chunk_size=chunks_size,
        chunk_overlap=chunk_overlap
    )
    chunked_doc=text_spliter.split_documents(documents)
    return chunked_doc

chunks=split_documents(all_text_doc)
print(len(chunks))

# convert chunks into embeddings
embedding_maneger=EmbeddingManeger()

# create vector store
vector_store=VectorStore()

# add documents
text=[doc.page_content for doc in chunks]
embedding=embedding_maneger.generate_embeddings(text)
vector_store.add_document(chunks,embedding)

# retrive documents
rag_retriver=RAGRetriver(embedding_maneger,vector_store)
rag_retriver.retrive("Veg Spring Roll")

# intigration with LLM
llm=ChatOllama(
    model="qwen2.5:3b",
    temperature=0.1,
    max_token=1024
)
def genetare_output(query,rag_retriver,llm,top_k=3,):
    result=rag_retriver.retrive(query,top_k)
    context="\n".join([doc["document"] for doc in result]) if result else ""
    if not context:
        print("We not found relevent contex for the given query")
    prompt=f""" Use given context to generate the answer for the given query
            context:{context}
            query:{query}
            """
    response=llm.invoke(prompt)
    return response.content

question=input("Enter your query: ")
answer=genetare_output(question,rag_retriver,llm)
print(answer)
