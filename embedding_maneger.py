from sentence_transformers import SentenceTransformer
class EmbeddingManeger:
    def __init__(self,model_name="all-MiniLM-L6-v2"):
        self.model_name=model_name
        print("loading model....")
        self.model=SentenceTransformer(self.model_name)
        print("Embedding dimension: ",self.model.get_embedding_dimension)
    def generate_embeddings(self,text):
        embeddings=self.model.encode(text)
        print("embedding shape: ",embeddings.shape)
        return embeddings
