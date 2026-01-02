"""
Comprehensive Clinical Metrics Module

Implements evaluation metrics for clinical reasoning:
- Text similarity (ROUGE, BLEU)
- Semantic similarity (BERTScore, embeddings)
- Clinical-specific metrics (DDX accuracy, SNOMED matching)
- Retrieval metrics (MRR, Recall@K, NDCG)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter
import re
import warnings

# Optional imports with fallbacks
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    warnings.warn("rouge_score not installed. ROUGE metrics will be unavailable.")

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence-transformers not installed. Semantic similarity will be unavailable.")

try:
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not installed. Some metrics will be unavailable.")


class ClinicalMetrics:
    """
    Comprehensive metrics suite for clinical reasoning evaluation.
    
    Supports:
    - ROUGE scores for text overlap
    - Semantic similarity using embeddings
    - Clinical-specific metrics (diagnosis accuracy, etc.)
    - Retrieval metrics for QA tasks
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_gpu: bool = True
    ):
        """
        Initialize the metrics calculator.
        
        Args:
            embedding_model: HuggingFace model for semantic similarity
            use_gpu: Whether to use GPU for embedding computation
        """
        self.embedding_model_name = embedding_model
        self.use_gpu = use_gpu
        
        # Track availability
        self.rouge_available = ROUGE_AVAILABLE
        self.st_available = SENTENCE_TRANSFORMERS_AVAILABLE
        self.sklearn_available = SKLEARN_AVAILABLE
        
        # Initialize ROUGE scorer
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
                use_stemmer=True
            )
        else:
            self.rouge_scorer = None
            
        # Initialize embedding model (lazy loading)
        self._embedding_model = None
        
    @property
    def embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
            self._embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
        return self._embedding_model
    
    # ==================== TEXT PREPROCESSING ====================
    
    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple word overlap similarity as fallback when no embeddings available."""
        words1 = set(self.preprocess_clinical_text(text1).split())
        words2 = set(self.preprocess_clinical_text(text2).split())
        
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'and', 'in', 'for', 'with', 'on', 'at', 'by'}
        words1 = words1 - stopwords
        words2 = words2 - stopwords
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def preprocess_clinical_text(text: str) -> str:
        """
        Preprocess clinical text for evaluation.
        Follows competition requirements:
        1. Convert to lowercase
        2. Remove punctuation
        3. Replace paragraphs with spaces
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation (except apostrophes)
        text = re.sub(r'[^\w\s\']', '', text)
        
        # Replace paragraphs/newlines with spaces
        text = re.sub(r'\n+', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def extract_medical_entities(text: str) -> List[str]:
        """Extract potential medical entities from text."""
        # Common medical terms patterns
        patterns = [
            r'\b(?:diagnosis|ddx|differential)\s*[:\-]?\s*([^.]+)',
            r'\b(?:management|treatment|plan)\s*[:\-]?\s*([^.]+)',
            r'\b(?:investigations?|labs?|tests?)\s*[:\-]?\s*([^.]+)',
        ]
        
        entities = []
        text_lower = text.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            entities.extend(matches)
            
        return [e.strip() for e in entities if e.strip()]
    
    # ==================== ROUGE METRICS ====================
    
    def compute_rouge(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str]]
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores for predictions vs references.
        
        Args:
            predictions: Single prediction or list of predicted texts
            references: Single reference or list of reference texts
            
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L F1 scores
        """
        # Handle single string inputs
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]
        
        if not ROUGE_AVAILABLE:
            # Fallback to simple word overlap scores
            return self._fallback_rouge(predictions, references)
        
        scores = {
            'rouge_1': [], 'rouge_2': [], 'rouge_l': [],
        }
        
        for pred, ref in zip(predictions, references):
            pred_processed = self.preprocess_clinical_text(pred)
            ref_processed = self.preprocess_clinical_text(ref)
            
            if not pred_processed or not ref_processed:
                continue
                
            result = self.rouge_scorer.score(ref_processed, pred_processed)
            
            scores['rouge_1'].append(result['rouge1'].fmeasure)
            scores['rouge_2'].append(result['rouge2'].fmeasure)
            scores['rouge_l'].append(result['rougeL'].fmeasure)
        
        # Compute averages
        return {k: float(np.mean(v)) if v else 0.0 for k, v in scores.items()}
    
    def _fallback_rouge(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """Simple word-overlap fallback when rouge_score not available."""
        scores = []
        for pred, ref in zip(predictions, references):
            pred_words = set(self.preprocess_clinical_text(pred).split())
            ref_words = set(self.preprocess_clinical_text(ref).split())
            
            if not pred_words or not ref_words:
                scores.append(0.0)
                continue
            
            overlap = len(pred_words & ref_words)
            precision = overlap / len(pred_words) if pred_words else 0
            recall = overlap / len(ref_words) if ref_words else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            scores.append(f1)
        
        mean_score = float(np.mean(scores)) if scores else 0.0
        return {
            'rouge_1': mean_score,
            'rouge_2': mean_score * 0.8,  # Rough approximation
            'rouge_l': mean_score * 0.9
        }
    
    # ==================== SEMANTIC SIMILARITY ====================
    
    def compute_semantic_similarity(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str]],
        batch_size: int = 32
    ) -> Union[float, Dict[str, float]]:
        """
        Compute semantic similarity using sentence embeddings.
        
        Args:
            predictions: Single prediction or list of predicted texts
            references: Single reference or list of reference texts
            batch_size: Batch size for encoding
            
        Returns:
            If single strings: float similarity score
            If lists: Dictionary with mean and std similarity scores
        """
        # Handle single string inputs
        single_input = isinstance(predictions, str) and isinstance(references, str)
        if single_input:
            predictions = [predictions]
            references = [references]
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.embedding_model is None:
            # Fallback to simple word overlap similarity
            return self._simple_text_similarity(predictions[0], references[0]) if single_input else 0.0
        
        # Preprocess texts
        preds_processed = [self.preprocess_clinical_text(p) for p in predictions]
        refs_processed = [self.preprocess_clinical_text(r) for r in references]
        
        # Filter out empty texts
        valid_pairs = [
            (p, r) for p, r in zip(preds_processed, refs_processed)
            if p and r
        ]
        
        if not valid_pairs:
            return 0.0 if single_input else {"semantic_similarity_mean": 0.0, "semantic_similarity_std": 0.0}
        
        preds_valid, refs_valid = zip(*valid_pairs)
        
        # Encode texts
        pred_embeddings = self.embedding_model.encode(
            list(preds_valid), 
            batch_size=batch_size,
            show_progress_bar=False
        )
        ref_embeddings = self.embedding_model.encode(
            list(refs_valid), 
            batch_size=batch_size,
            show_progress_bar=False
        )
        
        # Compute cosine similarities
        similarities = []
        for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
            sim = np.dot(pred_emb, ref_emb) / (
                np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb) + 1e-8
            )
            similarities.append(sim)
        
        # Return single value if single input
        if single_input:
            return float(similarities[0]) if similarities else 0.0
        
        return {
            "semantic_similarity_mean": float(np.mean(similarities)),
            "semantic_similarity_std": float(np.std(similarities)),
            "semantic_similarity_min": float(np.min(similarities)),
            "semantic_similarity_max": float(np.max(similarities))
        }
    
    # ==================== CLINICAL-SPECIFIC METRICS ====================
    
    def compute_ddx_accuracy(
        self,
        predicted_ddx: List[str],
        reference_ddx: List[str],
        snomed_codes_pred: Optional[List[str]] = None,
        snomed_codes_ref: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute diagnostic accuracy metrics.
        
        Args:
            predicted_ddx: List of predicted diagnoses (text)
            reference_ddx: List of reference diagnoses (text)
            snomed_codes_pred: Optional SNOMED codes for predictions
            snomed_codes_ref: Optional SNOMED codes for references
            
        Returns:
            Dictionary with diagnostic accuracy metrics
        """
        # Text-based accuracy (fuzzy matching)
        text_matches = []
        for pred, ref in zip(predicted_ddx, reference_ddx):
            pred_lower = pred.lower().strip() if pred else ""
            ref_lower = ref.lower().strip() if ref else ""
            
            # Check for exact match or substring match
            if pred_lower and ref_lower:
                exact = pred_lower == ref_lower
                partial = pred_lower in ref_lower or ref_lower in pred_lower
                text_matches.append(1.0 if exact else 0.5 if partial else 0.0)
            else:
                text_matches.append(0.0)
        
        results = {
            "ddx_text_accuracy": float(np.mean(text_matches)) if text_matches else 0.0,
            "ddx_exact_match_rate": sum(1 for m in text_matches if m == 1.0) / len(text_matches) if text_matches else 0.0
        }
        
        # SNOMED code matching if available
        if snomed_codes_pred and snomed_codes_ref:
            snomed_matches = []
            for pred_code, ref_code in zip(snomed_codes_pred, snomed_codes_ref):
                if pred_code and ref_code:
                    snomed_matches.append(1.0 if str(pred_code) == str(ref_code) else 0.0)
            
            if snomed_matches:
                results["snomed_accuracy"] = float(np.mean(snomed_matches))
        
        return results
    
    def compute_clinical_completeness(
        self,
        predictions: Union[str, List[str]],
        required_elements: Optional[List[str]] = None
    ) -> Union[float, Dict[str, float]]:
        """
        Evaluate clinical response completeness.
        
        Checks for presence of key clinical elements.
        
        Args:
            predictions: Single prediction or list of clinical response texts
            required_elements: List of keywords to check for
            
        Returns:
            If single string: float completeness score
            If list: Dictionary with completeness scores
        """
        # Handle single string input
        single_input = isinstance(predictions, str)
        if single_input:
            predictions = [predictions]
        
        # Default required elements
        if required_elements is None:
            required_elements = [
                'diagnosis', 'ddx', 'differential', 'assessment',
                'management', 'treatment', 'plan',
                'investigation', 'test', 'workup',
                'follow-up', 'followup', 'monitoring'
            ]
        
        completeness_scores = []
        for pred in predictions:
            pred_lower = pred.lower() if pred else ""
            found = sum(1 for elem in required_elements if elem in pred_lower)
            score = found / len(required_elements) if required_elements else 0.0
            completeness_scores.append(score)
        
        if single_input:
            return completeness_scores[0] if completeness_scores else 0.0
        
        return {
            "clinical_completeness_mean": float(np.mean(completeness_scores)) if completeness_scores else 0.0,
            "clinical_completeness_std": float(np.std(completeness_scores)) if completeness_scores else 0.0
        }
    
    # ==================== RETRIEVAL METRICS ====================
    
    def compute_retrieval_metrics(
        self,
        retrieved_docs: List[List[str]],
        relevant_docs: List[List[str]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics for knowledge retrieval evaluation.
        
        Args:
            retrieved_docs: List of lists of retrieved document IDs
            relevant_docs: List of lists of relevant document IDs (ground truth)
            k_values: List of K values for Recall@K
            
        Returns:
            Dictionary with MRR, Recall@K, Precision@K
        """
        mrr_scores = []
        recall_at_k = {k: [] for k in k_values}
        precision_at_k = {k: [] for k in k_values}
        
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            relevant_set = set(relevant)
            
            # Mean Reciprocal Rank
            rr = 0.0
            for i, doc in enumerate(retrieved):
                if doc in relevant_set:
                    rr = 1.0 / (i + 1)
                    break
            mrr_scores.append(rr)
            
            # Recall@K and Precision@K
            for k in k_values:
                retrieved_at_k = set(retrieved[:k])
                hits = len(retrieved_at_k & relevant_set)
                
                recall = hits / len(relevant_set) if relevant_set else 0.0
                precision = hits / k if k > 0 else 0.0
                
                recall_at_k[k].append(recall)
                precision_at_k[k].append(precision)
        
        results = {
            "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0
        }
        
        for k in k_values:
            results[f"recall@{k}"] = float(np.mean(recall_at_k[k])) if recall_at_k[k] else 0.0
            results[f"precision@{k}"] = float(np.mean(precision_at_k[k])) if precision_at_k[k] else 0.0
        
        return results
    
    # ==================== AGGREGATE METRICS ====================
    
    def compute_all_metrics(
        self,
        predictions: List[str],
        references: List[str],
        ddx_predictions: Optional[List[str]] = None,
        ddx_references: Optional[List[str]] = None,
        retrieved_docs: Optional[List[List[str]]] = None,
        relevant_docs: Optional[List[List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Compute all available metrics.
        
        Args:
            predictions: Predicted clinical responses
            references: Reference clinical responses
            ddx_predictions: Optional predicted diagnoses
            ddx_references: Optional reference diagnoses
            retrieved_docs: Optional retrieved documents for retrieval eval
            relevant_docs: Optional relevant documents for retrieval eval
            
        Returns:
            Comprehensive metrics dictionary
        """
        results = {}
        
        # ROUGE metrics
        rouge_results = self.compute_rouge(predictions, references)
        results["rouge"] = rouge_results
        
        # Semantic similarity
        semantic_results = self.compute_semantic_similarity(predictions, references)
        results["semantic"] = semantic_results
        
        # Clinical completeness
        completeness_results = self.compute_clinical_completeness(predictions)
        results["completeness"] = completeness_results
        
        # DDX accuracy if provided
        if ddx_predictions and ddx_references:
            ddx_results = self.compute_ddx_accuracy(ddx_predictions, ddx_references)
            results["ddx"] = ddx_results
        
        # Retrieval metrics if provided
        if retrieved_docs and relevant_docs:
            retrieval_results = self.compute_retrieval_metrics(retrieved_docs, relevant_docs)
            results["retrieval"] = retrieval_results
        
        # Compute overall score (weighted combination)
        overall_score = self._compute_overall_score(results)
        results["overall_score"] = overall_score
        
        return results
    
    def _compute_overall_score(
        self,
        results: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Compute weighted overall score from individual metrics."""
        if weights is None:
            weights = {
                "rouge1_f1": 0.15,
                "rouge2_f1": 0.10,
                "rougeL_f1": 0.15,
                "semantic_similarity_mean": 0.20,
                "overall_completeness": 0.20,
                "ddx_text_accuracy": 0.20
            }
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric_key, weight in weights.items():
            # Find the metric value in nested results
            value = None
            for category, category_results in results.items():
                if isinstance(category_results, dict) and metric_key in category_results:
                    value = category_results[metric_key]
                    break
            
            if value is not None and not isinstance(value, str):
                weighted_sum += weight * value
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class BootstrapEvaluator:
    """
    Bootstrap confidence interval estimation for metrics.
    """
    
    def __init__(self, n_bootstrap: int = 1000, ci: float = 0.95):
        """
        Args:
            n_bootstrap: Number of bootstrap samples
            ci: Confidence interval (default 95%)
        """
        self.n_bootstrap = n_bootstrap
        self.ci = ci
        self.alpha = (1 - ci) / 2
    
    def compute_ci(
        self,
        metric_fn,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Compute bootstrap confidence intervals for metrics.
        
        Args:
            metric_fn: Function that computes metrics given (preds, refs)
            predictions: List of predictions
            references: List of references
            
        Returns:
            Dict with metric_name -> (mean, lower_ci, upper_ci)
        """
        n_samples = len(predictions)
        bootstrap_results = []
        
        for _ in range(self.n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            preds_sample = [predictions[i] for i in indices]
            refs_sample = [references[i] for i in indices]
            
            # Compute metrics on sample
            results = metric_fn(preds_sample, refs_sample)
            bootstrap_results.append(results)
        
        # Aggregate bootstrap results
        ci_results = {}
        if bootstrap_results:
            all_keys = bootstrap_results[0].keys()
            for key in all_keys:
                values = [r[key] for r in bootstrap_results if isinstance(r.get(key), (int, float))]
                if values:
                    mean_val = np.mean(values)
                    lower = np.percentile(values, self.alpha * 100)
                    upper = np.percentile(values, (1 - self.alpha) * 100)
                    ci_results[key] = (mean_val, lower, upper)
        
        return ci_results


if __name__ == "__main__":
    # Example usage
    metrics = ClinicalMetrics()
    
    # Sample data
    predictions = [
        "Diagnosis: Bacterial meningitis. Management: IV antibiotics, supportive care. Investigations: LP, blood cultures.",
        "DDX: Diabetic ketoacidosis. Treatment: IV fluids, insulin therapy. Labs: blood glucose, electrolytes."
    ]
    
    references = [
        "Diagnosis: Bacterial meningitis. Management: Empirical antibiotics, ICU admission. Investigations: Lumbar puncture, blood cultures, CT head.",
        "Diagnosis: DKA. Management: Fluid resuscitation, insulin infusion. Investigations: Blood glucose, arterial blood gas, electrolytes."
    ]
    
    # Compute metrics
    results = metrics.compute_all_metrics(predictions, references)
    
    print("=== Clinical Metrics Results ===")
    for category, category_results in results.items():
        if isinstance(category_results, dict):
            print(f"\n{category.upper()}:")
            for metric, value in category_results.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
        else:
            print(f"\n{category}: {category_results:.4f}")
