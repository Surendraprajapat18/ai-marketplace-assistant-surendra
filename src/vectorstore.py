import os, pickle
import faiss
import numpy as np
from openai import OpenAI

class VectorStore:
    def __init__(self, index_dir, embed_model, openai_api_key):
        self.index_dir = index_dir
        self.embed_model = embed_model
        self.client = OpenAI(api_key=openai_api_key)
        self.index_path = os.path.join(index_dir, "faiss.index")
        self.meta_path = os.path.join(index_dir, "meta.pkl")
        self.index = None
        self.metas = []
        if os.path.exists(self.index_path):
            self._load()

    def _embed(self, texts):
        resp = self.client.embeddings.create(model=self.embed_model, input=texts)
        return np.array([d.embedding for d in resp.data], dtype="float32")

    def build_or_update(self, chunks, metas):
        embs = self._embed(chunks)
        faiss.normalize_L2(embs)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
        self.metas = [{**m, "text": t} for m, t in zip(metas, chunks)]
        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        pickle.dump(self.metas, open(self.meta_path, "wb"))

    def _load(self):
        self.index = faiss.read_index(self.index_path)
        self.metas = pickle.load(open(self.meta_path, "rb"))

    def search(self, query, top_k=4):
        q_emb = self._embed([query])
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb, top_k)
        return [
            {**self.metas[i], "score": float(scores[0][pos])}
            for pos, i in enumerate(idxs[0]) if i != -1
        ]
