import faiss
import numpy as np
import pickle
import os

class VectorDatabase:
    def __init__(self, dimension=384):  
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.load()

    def add(self, embedding, document):
        if self.index.ntotal == 0:
            self.index = faiss.IndexFlatL2(len(embedding))
        self.index.add(np.array([embedding], dtype=np.float32))
        self.documents.append(document)

    def search(self, query_embedding, k=5):
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)
        return [self.documents[i] for i in indices[0]]

    def save(self):
        faiss.write_index(self.index, "faiss_index.bin")
        with open("documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    def load(self):
        if os.path.exists("faiss_index.bin") and os.path.exists("documents.pkl"):
            self.index = faiss.read_index("faiss_index.bin")
            with open("documents.pkl", "rb") as f:
                self.documents = pickle.load(f)

    def clean(self):
        self.index.reset()
        self.documents = []


vector_db = VectorDatabase()
vector_db.clean()
