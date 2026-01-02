"""
Clinical Knowledge Retrieval Module

Implements retrieval-based evaluation following MedQA's IR approach:
- BM25/TF-IDF retrieval
- Semantic retrieval with embeddings
- Knowledge base management
- Retrieval-augmented generation evaluation
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from collections import defaultdict
import warnings

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not installed. TF-IDF retrieval unavailable.")

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence-transformers not installed. Semantic retrieval unavailable.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("faiss not installed. Fast nearest neighbor search unavailable.")


@dataclass
class Document:
    """Single document in the knowledge base."""
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    """Result from a retrieval query."""
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class ClinicalRetriever:
    """
    Multi-method clinical knowledge retriever.
    
    Supports:
    - TF-IDF based retrieval (sparse)
    - Semantic embedding retrieval (dense)
    - Hybrid retrieval (combination)
    - BM25-style scoring
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_gpu: bool = True,
        index_type: str = "flat"  # "flat" or "ivf" for FAISS
    ):
        """
        Initialize the retriever.
        
        Args:
            embedding_model: HuggingFace model for semantic retrieval
            use_gpu: Whether to use GPU for embeddings
            index_type: FAISS index type
        """
        self.embedding_model_name = embedding_model
        self.use_gpu = use_gpu
        self.index_type = index_type
        
        # Document storage
        self.documents: Dict[str, Document] = {}
        
        # TF-IDF components
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.doc_ids_order = []
        
        # Embedding components
        self._embedding_model = None
        self.embedding_index = None
        self.embedding_dim = None
        
    @property
    def embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
            self._embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
        return self._embedding_model
    
    # ==================== DOCUMENT MANAGEMENT ====================
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "text",
        id_field: str = "id"
    ):
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of document dictionaries
            text_field: Field containing document text
            id_field: Field containing document ID
        """
        for doc in documents:
            doc_id = str(doc.get(id_field, len(self.documents)))
            text = str(doc.get(text_field, ""))
            
            metadata = {k: v for k, v in doc.items() if k not in [text_field, id_field]}
            
            self.documents[doc_id] = Document(
                id=doc_id,
                text=text,
                metadata=metadata
            )
        
        # Rebuild indices
        self._build_tfidf_index()
    
    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None):
        """Add a single document."""
        self.documents[doc_id] = Document(
            id=doc_id,
            text=text,
            metadata=metadata or {}
        )
    
    def _preprocess_for_retrieval(self, text: str) -> str:
        """Preprocess text for retrieval."""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # ==================== TF-IDF RETRIEVAL ====================
    
    def _build_tfidf_index(self):
        """Build TF-IDF index from documents."""
        if not SKLEARN_AVAILABLE or not self.documents:
            return
        
        self.doc_ids_order = list(self.documents.keys())
        texts = [
            self._preprocess_for_retrieval(self.documents[doc_id].text)
            for doc_id in self.doc_ids_order
        ]
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True  # Use BM25-like sublinear TF scaling
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
    
    def retrieve_tfidf(
        self,
        query: str,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using TF-IDF similarity.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of RetrievalResult objects
        """
        if not SKLEARN_AVAILABLE or self.tfidf_vectorizer is None:
            return []
        
        query_processed = self._preprocess_for_retrieval(query)
        query_vector = self.tfidf_vectorizer.transform([query_processed])
        
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc_id = self.doc_ids_order[idx]
            doc = self.documents[doc_id]
            results.append(RetrievalResult(
                doc_id=doc_id,
                text=doc.text,
                score=float(similarities[idx]),
                metadata=doc.metadata
            ))
        
        return results
    
    # ==================== SEMANTIC RETRIEVAL ====================
    
    def build_embedding_index(self, batch_size: int = 32):
        """Build semantic embedding index."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.embedding_model is None:
            warnings.warn("Cannot build embedding index without sentence-transformers")
            return
        
        self.doc_ids_order = list(self.documents.keys())
        texts = [self.documents[doc_id].text for doc_id in self.doc_ids_order]
        
        # Encode all documents
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Store embeddings in documents
        for i, doc_id in enumerate(self.doc_ids_order):
            self.documents[doc_id].embedding = embeddings[i]
        
        self.embedding_dim = embeddings.shape[1]
        
        # Build FAISS index if available
        if FAISS_AVAILABLE:
            if self.index_type == "flat":
                self.embedding_index = faiss.IndexFlatIP(self.embedding_dim)
            elif self.index_type == "ivf":
                nlist = min(100, len(embeddings) // 10)
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                self.embedding_index = faiss.IndexIVFFlat(
                    quantizer, self.embedding_dim, nlist
                )
                self.embedding_index.train(embeddings.astype('float32'))
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings.astype('float32'))
            self.embedding_index.add(embeddings.astype('float32'))
        else:
            # Store as numpy array for brute force search
            self.embedding_index = embeddings
    
    def retrieve_semantic(
        self,
        query: str,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using semantic similarity.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of RetrievalResult objects
        """
        if self.embedding_model is None or self.embedding_index is None:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        
        if FAISS_AVAILABLE and hasattr(self.embedding_index, 'search'):
            # Normalize query for cosine similarity
            query_norm = query_embedding.astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_norm)
            
            scores, indices = self.embedding_index.search(query_norm, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Brute force with numpy
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                self.embedding_index
            )[0]
            indices = np.argsort(similarities)[::-1][:top_k]
            scores = similarities[indices]
        
        results = []
        for idx, score in zip(indices, scores):
            if idx < len(self.doc_ids_order):
                doc_id = self.doc_ids_order[idx]
                doc = self.documents[doc_id]
                results.append(RetrievalResult(
                    doc_id=doc_id,
                    text=doc.text,
                    score=float(score),
                    metadata=doc.metadata
                ))
        
        return results
    
    # ==================== HYBRID RETRIEVAL ====================
    
    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        tfidf_weight: float = 0.3,
        semantic_weight: float = 0.7
    ) -> List[RetrievalResult]:
        """
        Hybrid retrieval combining TF-IDF and semantic methods.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            tfidf_weight: Weight for TF-IDF scores
            semantic_weight: Weight for semantic scores
            
        Returns:
            List of RetrievalResult objects with combined scores
        """
        # Get more candidates from each method
        n_candidates = top_k * 3
        
        tfidf_results = self.retrieve_tfidf(query, n_candidates)
        semantic_results = self.retrieve_semantic(query, n_candidates)
        
        # Combine scores
        doc_scores = defaultdict(lambda: {'tfidf': 0.0, 'semantic': 0.0, 'doc': None})
        
        # Normalize TF-IDF scores
        if tfidf_results:
            max_tfidf = max(r.score for r in tfidf_results)
            for r in tfidf_results:
                normalized_score = r.score / max_tfidf if max_tfidf > 0 else 0
                doc_scores[r.doc_id]['tfidf'] = normalized_score
                doc_scores[r.doc_id]['doc'] = r
        
        # Normalize semantic scores
        if semantic_results:
            max_semantic = max(r.score for r in semantic_results)
            for r in semantic_results:
                normalized_score = r.score / max_semantic if max_semantic > 0 else 0
                doc_scores[r.doc_id]['semantic'] = normalized_score
                if doc_scores[r.doc_id]['doc'] is None:
                    doc_scores[r.doc_id]['doc'] = r
        
        # Compute combined scores
        combined_results = []
        for doc_id, scores in doc_scores.items():
            combined_score = (
                tfidf_weight * scores['tfidf'] +
                semantic_weight * scores['semantic']
            )
            result = scores['doc']
            if result:
                combined_results.append(RetrievalResult(
                    doc_id=result.doc_id,
                    text=result.text,
                    score=combined_score,
                    metadata=result.metadata
                ))
        
        # Sort by combined score and return top-k
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:top_k]
    
    # ==================== RETRIEVAL EVALUATION ====================
    
    def evaluate_retrieval(
        self,
        queries: List[str],
        relevant_docs: List[List[str]],
        method: str = "hybrid",
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Args:
            queries: List of query texts
            relevant_docs: List of lists of relevant doc IDs for each query
            method: "tfidf", "semantic", or "hybrid"
            k_values: K values for Recall@K
            
        Returns:
            Dictionary with MRR, Recall@K metrics
        """
        retrieve_fn = {
            "tfidf": self.retrieve_tfidf,
            "semantic": self.retrieve_semantic,
            "hybrid": self.retrieve_hybrid
        }.get(method, self.retrieve_hybrid)
        
        mrr_scores = []
        recall_at_k = {k: [] for k in k_values}
        
        max_k = max(k_values)
        
        for query, relevant in zip(queries, relevant_docs):
            results = retrieve_fn(query, top_k=max_k)
            retrieved_ids = [r.doc_id for r in results]
            relevant_set = set(relevant)
            
            # MRR
            rr = 0.0
            for i, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_set:
                    rr = 1.0 / (i + 1)
                    break
            mrr_scores.append(rr)
            
            # Recall@K
            for k in k_values:
                retrieved_at_k = set(retrieved_ids[:k])
                hits = len(retrieved_at_k & relevant_set)
                recall = hits / len(relevant_set) if relevant_set else 0.0
                recall_at_k[k].append(recall)
        
        results = {"mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0}
        for k in k_values:
            results[f"recall@{k}"] = float(np.mean(recall_at_k[k])) if recall_at_k[k] else 0.0
        
        return results


class ClinicalKnowledgeBase:
    """
    Structured clinical knowledge base for Kenya-specific medical knowledge.
    
    Categories:
    - Clinical guidelines (Kenya MOH)
    - Drug information
    - Disease presentations
    - Management protocols
    """
    
    def __init__(self):
        self.categories = {
            'guidelines': [],
            'drugs': [],
            'diseases': [],
            'protocols': [],
            'general': []
        }
        self.retriever = ClinicalRetriever()
    
    def add_knowledge(
        self,
        text: str,
        category: str,
        source: str,
        metadata: Optional[Dict] = None
    ):
        """Add knowledge to the base."""
        doc_id = f"{category}_{len(self.categories.get(category, []))}"
        
        full_metadata = {
            'category': category,
            'source': source,
            **(metadata or {})
        }
        
        self.categories.setdefault(category, []).append({
            'id': doc_id,
            'text': text,
            **full_metadata
        })
        
        self.retriever.add_document(doc_id, text, full_metadata)
    
    def build_from_responses(self, df, response_columns: List[str]):
        """
        Build knowledge base from clinical response columns.
        
        Args:
            df: DataFrame with clinical responses
            response_columns: Columns containing responses (e.g., ['Clinician', 'GPT4.0'])
        """
        for idx, row in df.iterrows():
            for col in response_columns:
                response = str(row.get(col, ''))
                if response and len(response) > 50:
                    self.add_knowledge(
                        text=response,
                        category='general',
                        source=col,
                        metadata={
                            'clinical_panel': row.get('Clinical Panel', ''),
                            'county': row.get('County', ''),
                            'original_idx': idx
                        }
                    )
        
        # Build retrieval indices
        self.retriever._build_tfidf_index()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        method: str = "hybrid"
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant knowledge.
        
        Args:
            query: Query text
            top_k: Number of results
            category: Optional category filter
            method: Retrieval method
        """
        results = self.retriever.retrieve_hybrid(query, top_k * 2)
        
        if category:
            results = [r for r in results if r.metadata.get('category') == category]
        
        return results[:top_k]


if __name__ == "__main__":
    # Example usage
    retriever = ClinicalRetriever()
    
    # Sample documents
    docs = [
        {"id": "doc1", "text": "Management of diabetic ketoacidosis involves IV fluids, insulin therapy, and electrolyte monitoring.", "category": "protocols"},
        {"id": "doc2", "text": "Bacterial meningitis requires empirical antibiotics, lumbar puncture for diagnosis, and ICU admission.", "category": "protocols"},
        {"id": "doc3", "text": "Malaria treatment in Kenya follows WHO guidelines with artemisinin-based combination therapy.", "category": "guidelines"},
        {"id": "doc4", "text": "Hypertensive emergency management includes IV labetalol or nitroprusside with continuous monitoring.", "category": "protocols"},
        {"id": "doc5", "text": "Pediatric pneumonia assessment includes respiratory rate, chest indrawing, and oxygen saturation.", "category": "guidelines"},
    ]
    
    retriever.add_documents(docs)
    
    # Test TF-IDF retrieval
    print("=== TF-IDF Retrieval ===")
    results = retriever.retrieve_tfidf("How to manage diabetic patient with high blood sugar?", top_k=3)
    for r in results:
        print(f"  {r.doc_id}: {r.score:.4f} - {r.text[:50]}...")
    
    print("\n=== Evaluation Example ===")
    queries = ["diabetic ketoacidosis management", "meningitis treatment"]
    relevant = [["doc1"], ["doc2"]]
    
    eval_results = retriever.evaluate_retrieval(queries, relevant, method="tfidf")
    print(f"MRR: {eval_results['mrr']:.4f}")
    for k in [1, 3, 5]:
        print(f"Recall@{k}: {eval_results.get(f'recall@{k}', 0):.4f}")
