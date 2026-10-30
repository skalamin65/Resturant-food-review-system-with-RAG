class RAGRetriver:
    def __init__(self,embedding_maneger,vectore_store):
        self.embedding_maneger=embedding_maneger
        self.vectore_store=vectore_store
    def retrive(self,query,top_k=5,score_threshold=0.0):
        # query
        query_embeddings=self.embedding_maneger.generate_embeddings([query])[0]

        # results
        results=self.vectore_store.collection.query(
            query_embeddings=[query_embeddings.tolist()],
            n_results=top_k
        )

        # cosine similarity
        retrived_docs=[]
        if results["documents"] and results["documents"][0]:
            ids=results["ids"]
            metadatas=results["metadatas"][0]
            documents=results["documents"][0]
            distances=results["distances"][0]

            for i,(doc_id,metadata,document,distance) in enumerate(zip(ids,metadatas,documents,distances)):
                simiarity_Score=1-distance

                if simiarity_Score>=score_threshold:
                    retrived_docs.append({
                        "id":doc_id,
                        "metadata":metadata,
                        "document":document,
                        "distance":distance,
                        "similarity_score":simiarity_Score,
                        "rank":i+1
                    })
                    print(f"retrive {len(retrived_docs)} documents")
        else:
            print("No documents found!")
        return retrived_docs