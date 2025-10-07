"""
FAISS index wrapper with fallbacks and utilities.

Provides a small API around faiss/hnswlib or numpy fallback to build, save, load and search
normalized float32 embeddings. Saves metadata files alongside index.
"""
import os
import json
import pickle
import numpy as np
import logging

logger = logging.getLogger(__name__)

_HAS_FAISS = False
_HAS_HNSW = False
try:
    import faiss
    _HAS_FAISS = True
    logger.info('faiss available')
except Exception:
    try:
        import hnswlib
        _HAS_HNSW = True
        logger.info('hnswlib available')
    except Exception:
        logger.info('No faiss/hnswlib available, using numpy fallback')


class FaissIndex:
    def __init__(self, dim: int = None):
        self.dim = dim
        self.index = None
        self.names = []
        self.embeddings = None

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def build(self, embeddings: np.ndarray, names: list):
        """Build index from embeddings (N, dim) and names list"""
        if embeddings is None or len(embeddings) == 0:
            raise ValueError('No embeddings provided')
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError('Embeddings must be 2D')
        self.dim = embeddings.shape[1]
        embeddings = self._normalize(embeddings)
        self.embeddings = embeddings
        self.names = list(names)

        if _HAS_FAISS:
            # Use inner product on normalized vectors = cosine similarity
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(embeddings)
        elif _HAS_HNSW:
            self.index = hnswlib.Index(space='cosine', dim=self.dim)
            self.index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
            self.index.add_items(embeddings, np.arange(len(embeddings)))
            self.index.set_ef(50)
        else:
            # numpy fallback: store embeddings and use dot product
            self.index = None

    def add(self, new_embeddings: np.ndarray, new_names: list):
        if new_embeddings is None or len(new_embeddings) == 0:
            return
        new_embeddings = np.asarray(new_embeddings, dtype=np.float32)
        if new_embeddings.ndim == 1:
            new_embeddings = new_embeddings.reshape(1, -1)
        if self.dim is None:
            self.build(new_embeddings, new_names)
            return
        if new_embeddings.shape[1] != self.dim:
            raise ValueError('Dimension mismatch')
        new_embeddings = self._normalize(new_embeddings)
        if _HAS_FAISS:
            self.index.add(new_embeddings)
        elif _HAS_HNSW:
            start_id = len(self.names)
            self.index.add_items(new_embeddings, np.arange(start_id, start_id + len(new_embeddings)))
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings]) if self.embeddings is not None else new_embeddings
        self.names.extend(list(new_names))

    def search(self, query: np.ndarray, k: int = 1):
        """Search for nearest neighbors. Returns (distances, indices, names)
        distances: similarity scores (for faiss IndexFlatIP higher is better), numpy array shape (1,k)
        indices: indices into names list
        names: corresponding names
        """
        q = np.asarray(query, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        # normalize
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-10)

        if _HAS_FAISS and isinstance(self.index, faiss.Index):
            D, I = self.index.search(q, k)
            # D is inner-product similarity
            names = [[self.names[int(idx)] if idx >= 0 and idx < len(self.names) else None for idx in row] for row in I]
            return D, I, names
        elif _HAS_HNSW and self.index is not None:
            labels, distances = self.index.knn_query(q, k=k)
            # hnswlib returns labels and cosine distances; convert to similarity
            sim = 1.0 - distances
            names = [[self.names[int(idx)] if idx >= 0 and idx < len(self.names) else None for idx in row] for row in labels]
            return sim, labels, names
        else:
            # brute force dot product
            if self.embeddings is None or len(self.embeddings) == 0:
                return np.zeros((q.shape[0], k)), -1 * np.ones((q.shape[0], k), dtype=int), [[None]*k for _ in range(q.shape[0])]
            sims = np.dot(q, self.embeddings.T)
            idx = np.argsort(-sims, axis=1)[:, :k]
            top = np.take_along_axis(sims, idx, axis=1)
            names = [[self.names[int(j)] for j in row] for row in idx]
            return top, idx, names

    def save(self, base_path: str):
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        # save metadata
        meta_path = base_path + '.meta.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({'dim': self.dim, 'names': self.names}, f)
        # save embeddings/names as pickle
        pkl_path = base_path + '.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump({'embeddings': self.embeddings, 'names': self.names}, f)
        # save index
        if _HAS_FAISS and isinstance(self.index, faiss.Index):
            faiss.write_index(self.index, base_path + '.index')
        elif _HAS_HNSW and self.index is not None:
            self.index.save_index(base_path + '.hnsw')
        else:
            # numpy fallback already saved in pkl
            pass

    def load(self, base_path: str):
        meta_path = base_path + '.meta.json'
        pkl_path = base_path + '.pkl'
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                self.dim = int(meta.get('dim'))
                self.names = meta.get('names', [])
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data.get('embeddings')
                if self.embeddings is not None:
                    self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
        if _HAS_FAISS and os.path.exists(base_path + '.index'):
            self.index = faiss.read_index(base_path + '.index')
        elif _HAS_HNSW and os.path.exists(base_path + '.hnsw'):
            import hnswlib
            self.index = hnswlib.Index(space='cosine', dim=self.dim)
            self.index.load_index(base_path + '.hnsw')
        else:
            # nothing to do for numpy fallback - embeddings already loaded
            pass


def quick_test():
    # small smoke test
    idx = FaissIndex()
    embs = np.random.randn(10, 512).astype(np.float32)
    names = [f's{i}' for i in range(10)]
    idx.build(embs, names)
    q = embs[3] + 0.001
    D, I, N = idx.search(q, k=3)
    print('quick_test:', D.shape, I.shape)


if __name__ == '__main__':
    quick_test()
