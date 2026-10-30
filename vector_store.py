import os
import chromadb
import uuid
class VectorStore:
    def __init__(self,persist_directory="Data/vector_store",collection_name="restaurant_documents"):
        self.persist_directory=persist_directory
        self.collection_name=collection_name
        self.client=None
        self.collection=None

        self._initialize()
    def _initialize(self):
        os.makedirs(self.persist_directory,exist_ok=True)
        self.client=chromadb.PersistentClient(path=self.persist_directory)
        self.collection=self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description":"vector store for RAG pipeline"}
        )
        print("Initialize vector store with : ",self.collection_name)
        print("docs in collection: ", self.collection.count())

    def add_document(self,documents,embeddings):
        if len(documents)!=len(embeddings):
            raise ValueError("Number of documents doesn't match the number of embeddings")
        # store => ids,embedding,document,metadata
        ids=[]
        embeddings_list=[]
        documents_content=[]
        all_metadata=[]

        for i,(doc,embedding) in enumerate(zip(documents,embeddings)):
            doc_id=f"doc_{uuid.uuid4()}"
            ids.append(doc_id)

            metadata=dict(doc.metadata)
            metadata["document_index"]=i
            metadata["content_length"]=len(doc.page_content)
            all_metadata.append(metadata)

            documents_content.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents_content,
                metadatas=all_metadata
            )
        print("Total documents added in vector store: ",len(documents_content))
        print("doc in collection: ",self.collection.count())