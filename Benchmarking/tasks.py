"""
Clinical Benchmark Tasks Module

Defines task-specific evaluators following standard ML benchmark patterns:
- Clinical Reasoning Task (free-text generation)
- Diagnostic Task (DDX prediction)
- Multi-Choice Task (MedQA-style)
- Retrieval QA Task (knowledge-grounded)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
import re

from .datasets import TaskType, BenchmarkSample, BenchmarkDataset
from .metrics import ClinicalMetrics


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskResult:
    """Result from evaluating a single sample."""
    sample_id: str
    prediction: str
    ground_truth: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskEvaluationResult:
    """Aggregated results for an entire task."""
    task_name: str
    task_type: TaskType
    num_samples: int
    aggregate_metrics: Dict[str, float]
    per_sample_results: List[TaskResult]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTask(ABC):
    """Abstract base class for benchmark tasks."""
    
    def __init__(
        self,
        name: str,
        task_type: TaskType,
        metrics: Optional[ClinicalMetrics] = None
    ):
        self.name = name
        self.task_type = task_type
        self.metrics = metrics or ClinicalMetrics()
        self.status = TaskStatus.PENDING
    
    @abstractmethod
    def evaluate_sample(
        self,
        sample: BenchmarkSample,
        prediction: str
    ) -> TaskResult:
        """Evaluate a single sample."""
        pass
    
    @abstractmethod
    def evaluate_dataset(
        self,
        dataset: BenchmarkDataset,
        predictions: List[str]
    ) -> TaskEvaluationResult:
        """Evaluate an entire dataset."""
        pass
    
    def get_prompt_template(self) -> str:
        """Get the prompt template for this task."""
        return "{input}"


class ClinicalReasoningTask(BaseTask):
    """
    Free-text clinical reasoning evaluation.
    
    Evaluates:
    - ROUGE scores against clinician reference
    - Semantic similarity
    - Clinical completeness
    """
    
    def __init__(
        self,
        name: str = "clinical_reasoning",
        metrics: Optional[ClinicalMetrics] = None,
        completeness_keywords: Optional[List[str]] = None
    ):
        super().__init__(name, TaskType.CLINICAL_REASONING, metrics)
        self.completeness_keywords = completeness_keywords or [
            'assessment', 'diagnosis', 'differential', 'management',
            'treatment', 'investigation', 'examination', 'history',
            'symptoms', 'signs', 'prognosis', 'follow-up', 'referral'
        ]
    
    def get_prompt_template(self) -> str:
        return """You are an expert clinical consultant. Based on the following clinical scenario, provide a comprehensive clinical reasoning and management plan.

CLINICAL SCENARIO:
{input}

Provide your response covering:
1. Initial Assessment
2. Differential Diagnosis (DDX)
3. Recommended Investigations
4. Management Plan
5. Follow-up Recommendations

RESPONSE:"""
    
    def evaluate_sample(
        self,
        sample: BenchmarkSample,
        prediction: str
    ) -> TaskResult:
        """Evaluate a single clinical reasoning sample."""
        ground_truth = sample.expected_output
        
        # Compute metrics
        rouge_scores = self.metrics.compute_rouge(prediction, ground_truth)
        semantic_sim = self.metrics.compute_semantic_similarity(prediction, ground_truth)
        completeness = self.metrics.compute_clinical_completeness(
            prediction, self.completeness_keywords
        )
        
        metrics = {
            **rouge_scores,
            'semantic_similarity': semantic_sim,
            'clinical_completeness': completeness
        }
        
        return TaskResult(
            sample_id=sample.id,
            prediction=prediction,
            ground_truth=ground_truth,
            metrics=metrics,
            metadata=sample.metadata
        )
    
    def evaluate_dataset(
        self,
        dataset: BenchmarkDataset,
        predictions: List[str]
    ) -> TaskEvaluationResult:
        """Evaluate entire dataset for clinical reasoning."""
        self.status = TaskStatus.RUNNING
        
        per_sample_results = []
        for sample, pred in zip(dataset.samples, predictions):
            result = self.evaluate_sample(sample, pred)
            per_sample_results.append(result)
        
        # Aggregate metrics
        aggregate_metrics = self._aggregate_metrics(per_sample_results)
        
        self.status = TaskStatus.COMPLETED
        
        return TaskEvaluationResult(
            task_name=self.name,
            task_type=self.task_type,
            num_samples=len(dataset.samples),
            aggregate_metrics=aggregate_metrics,
            per_sample_results=per_sample_results,
            metadata={'dataset_name': dataset.name}
        )
    
    def _aggregate_metrics(self, results: List[TaskResult]) -> Dict[str, float]:
        """Aggregate metrics across all samples."""
        if not results:
            return {}
        
        metric_names = list(results[0].metrics.keys())
        aggregated = {}
        
        for metric in metric_names:
            values = [r.metrics.get(metric, 0.0) for r in results]
            aggregated[f"{metric}_mean"] = float(np.mean(values))
            aggregated[f"{metric}_std"] = float(np.std(values))
        
        return aggregated


class DiagnosticTask(BaseTask):
    """
    Differential diagnosis (DDX) prediction task.
    
    Evaluates:
    - DDX accuracy (exact match, partial match)
    - Top-K accuracy
    - SNOMED code matching
    """
    
    def __init__(
        self,
        name: str = "diagnostic",
        metrics: Optional[ClinicalMetrics] = None,
        top_k: int = 5
    ):
        super().__init__(name, TaskType.DIAGNOSTIC, metrics)
        self.top_k = top_k
    
    def get_prompt_template(self) -> str:
        return """Based on the following clinical presentation, provide your differential diagnosis list ranked by likelihood.

CLINICAL PRESENTATION:
{input}

Provide your differential diagnoses in the following format:
1. [Most likely diagnosis]
2. [Second most likely]
3. [Third most likely]
...

DIFFERENTIAL DIAGNOSIS:"""
    
    def _parse_ddx_list(self, text: str) -> List[str]:
        """Parse differential diagnosis list from text."""
        # Try numbered list format
        pattern = r'^\d+[\.\)]\s*(.+?)$'
        matches = re.findall(pattern, text, re.MULTILINE)
        
        if matches:
            return [m.strip() for m in matches if m.strip()]
        
        # Try bullet points
        lines = text.split('\n')
        ddx_list = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove common prefixes
                line = re.sub(r'^[\-\*•]\s*', '', line)
                if line:
                    ddx_list.append(line)
        
        return ddx_list[:self.top_k * 2]  # Cap at reasonable number
    
    def evaluate_sample(
        self,
        sample: BenchmarkSample,
        prediction: str
    ) -> TaskResult:
        """Evaluate a single diagnostic sample."""
        ground_truth = sample.expected_output
        
        # Parse DDX lists
        pred_ddx = self._parse_ddx_list(prediction)
        true_ddx = self._parse_ddx_list(ground_truth)
        
        # Get SNOMED codes if available
        snomed_codes = sample.metadata.get('ddx_snomed', [])
        
        # Compute DDX accuracy
        ddx_metrics = self.metrics.compute_ddx_accuracy(pred_ddx, true_ddx)
        
        # Compute semantic similarity as backup
        semantic_sim = self.metrics.compute_semantic_similarity(prediction, ground_truth)
        
        metrics = {
            **ddx_metrics,
            'semantic_similarity': semantic_sim,
            'num_predicted_ddx': len(pred_ddx),
            'num_true_ddx': len(true_ddx)
        }
        
        return TaskResult(
            sample_id=sample.id,
            prediction=prediction,
            ground_truth=ground_truth,
            metrics=metrics,
            metadata={
                **sample.metadata,
                'predicted_ddx': pred_ddx,
                'true_ddx': true_ddx
            }
        )
    
    def evaluate_dataset(
        self,
        dataset: BenchmarkDataset,
        predictions: List[str]
    ) -> TaskEvaluationResult:
        """Evaluate entire dataset for diagnostics."""
        self.status = TaskStatus.RUNNING
        
        per_sample_results = []
        for sample, pred in zip(dataset.samples, predictions):
            result = self.evaluate_sample(sample, pred)
            per_sample_results.append(result)
        
        aggregate_metrics = self._aggregate_metrics(per_sample_results)
        
        self.status = TaskStatus.COMPLETED
        
        return TaskEvaluationResult(
            task_name=self.name,
            task_type=self.task_type,
            num_samples=len(dataset.samples),
            aggregate_metrics=aggregate_metrics,
            per_sample_results=per_sample_results,
            metadata={'dataset_name': dataset.name}
        )
    
    def _aggregate_metrics(self, results: List[TaskResult]) -> Dict[str, float]:
        """Aggregate diagnostic metrics."""
        if not results:
            return {}
        
        metric_names = [
            'exact_match', 'partial_match', 'top1_accuracy',
            'top3_accuracy', 'top5_accuracy', 'semantic_similarity'
        ]
        
        aggregated = {}
        for metric in metric_names:
            values = [r.metrics.get(metric, 0.0) for r in results if metric in r.metrics]
            if values:
                aggregated[f"{metric}_mean"] = float(np.mean(values))
                aggregated[f"{metric}_std"] = float(np.std(values))
        
        return aggregated


class MultiChoiceTask(BaseTask):
    """
    Multiple choice QA task (MedQA style).
    
    Evaluates:
    - Accuracy (correct choice selection)
    - Calibration (confidence vs accuracy)
    """
    
    def __init__(
        self,
        name: str = "multi_choice",
        metrics: Optional[ClinicalMetrics] = None
    ):
        super().__init__(name, TaskType.MULTI_CHOICE, metrics)
    
    def get_prompt_template(self) -> str:
        return """Answer the following clinical question by selecting the most appropriate option.

QUESTION:
{input}

OPTIONS:
{choices}

Select your answer (A, B, C, D, or E):"""
    
    def format_prompt(self, sample: BenchmarkSample) -> str:
        """Format prompt with choices."""
        template = self.get_prompt_template()
        
        choices = sample.metadata.get('choices', [])
        choices_text = "\n".join([
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)
        ])
        
        return template.format(
            input=sample.input_text,
            choices=choices_text
        )
    
    def _parse_answer(self, text: str) -> str:
        """Extract answer letter from response."""
        text = text.strip().upper()
        
        # Direct letter answer
        if text and text[0] in 'ABCDE':
            return text[0]
        
        # Find letter in text
        match = re.search(r'\b([A-E])\b', text)
        if match:
            return match.group(1)
        
        # Check for written answers
        answer_patterns = {
            'A': ['answer a', 'option a', 'choice a'],
            'B': ['answer b', 'option b', 'choice b'],
            'C': ['answer c', 'option c', 'choice c'],
            'D': ['answer d', 'option d', 'choice d'],
            'E': ['answer e', 'option e', 'choice e'],
        }
        
        text_lower = text.lower()
        for letter, patterns in answer_patterns.items():
            if any(p in text_lower for p in patterns):
                return letter
        
        return ""
    
    def evaluate_sample(
        self,
        sample: BenchmarkSample,
        prediction: str
    ) -> TaskResult:
        """Evaluate a single multi-choice sample."""
        correct_answer = sample.metadata.get('correct_answer', '')
        
        predicted_answer = self._parse_answer(prediction)
        is_correct = predicted_answer == correct_answer.upper()
        
        metrics = {
            'accuracy': 1.0 if is_correct else 0.0,
            'answer_parsed': 1.0 if predicted_answer else 0.0
        }
        
        return TaskResult(
            sample_id=sample.id,
            prediction=predicted_answer,
            ground_truth=correct_answer,
            metrics=metrics,
            metadata={
                **sample.metadata,
                'raw_prediction': prediction,
                'is_correct': is_correct
            }
        )
    
    def evaluate_dataset(
        self,
        dataset: BenchmarkDataset,
        predictions: List[str]
    ) -> TaskEvaluationResult:
        """Evaluate entire dataset for multi-choice."""
        self.status = TaskStatus.RUNNING
        
        per_sample_results = []
        for sample, pred in zip(dataset.samples, predictions):
            result = self.evaluate_sample(sample, pred)
            per_sample_results.append(result)
        
        # Calculate accuracy
        correct = sum(1 for r in per_sample_results if r.metrics.get('accuracy', 0) == 1.0)
        total = len(per_sample_results)
        
        aggregate_metrics = {
            'accuracy': correct / total if total > 0 else 0.0,
            'num_correct': correct,
            'num_total': total,
            'parse_rate': sum(r.metrics.get('answer_parsed', 0) for r in per_sample_results) / total if total > 0 else 0.0
        }
        
        self.status = TaskStatus.COMPLETED
        
        return TaskEvaluationResult(
            task_name=self.name,
            task_type=self.task_type,
            num_samples=total,
            aggregate_metrics=aggregate_metrics,
            per_sample_results=per_sample_results,
            metadata={'dataset_name': dataset.name}
        )


class RetrievalQATask(BaseTask):
    """
    Retrieval-augmented QA task.
    
    Evaluates:
    - Answer quality with retrieval context
    - Retrieval quality metrics
    - Attribution/grounding scores
    """
    
    def __init__(
        self,
        name: str = "retrieval_qa",
        metrics: Optional[ClinicalMetrics] = None,
        retriever: Optional[Any] = None  # ClinicalRetriever instance
    ):
        super().__init__(name, TaskType.RETRIEVAL_QA, metrics)
        self.retriever = retriever
    
    def get_prompt_template(self) -> str:
        return """Answer the following clinical question using the provided context.

CONTEXT:
{context}

QUESTION:
{input}

Provide a comprehensive answer based on the context provided:"""
    
    def format_prompt(self, sample: BenchmarkSample, retrieved_context: str = "") -> str:
        """Format prompt with retrieved context."""
        template = self.get_prompt_template()
        
        context = retrieved_context or sample.metadata.get('context', '')
        
        return template.format(
            input=sample.input_text,
            context=context
        )
    
    def evaluate_sample(
        self,
        sample: BenchmarkSample,
        prediction: str,
        retrieved_docs: Optional[List[str]] = None
    ) -> TaskResult:
        """Evaluate a single retrieval QA sample."""
        ground_truth = sample.expected_output
        
        # Answer quality metrics
        rouge_scores = self.metrics.compute_rouge(prediction, ground_truth)
        semantic_sim = self.metrics.compute_semantic_similarity(prediction, ground_truth)
        
        metrics = {
            **rouge_scores,
            'semantic_similarity': semantic_sim,
        }
        
        # Attribution scoring (if context available)
        context = sample.metadata.get('context', '')
        if context:
            attribution_score = self._compute_attribution(prediction, context)
            metrics['attribution_score'] = attribution_score
        
        # Retrieval metrics (if ground truth docs available)
        if retrieved_docs and 'relevant_docs' in sample.metadata:
            relevant_docs = sample.metadata['relevant_docs']
            retrieval_metrics = self.metrics.compute_retrieval_metrics(
                retrieved_docs, relevant_docs, k_values=[1, 3, 5]
            )
            metrics.update(retrieval_metrics)
        
        return TaskResult(
            sample_id=sample.id,
            prediction=prediction,
            ground_truth=ground_truth,
            metrics=metrics,
            metadata={
                **sample.metadata,
                'retrieved_docs': retrieved_docs
            }
        )
    
    def _compute_attribution(self, answer: str, context: str) -> float:
        """
        Compute how well the answer is grounded in the context.
        Simple overlap-based scoring.
        """
        answer_tokens = set(answer.lower().split())
        context_tokens = set(context.lower().split())
        
        # Remove stopwords (simple list)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'and', 'in', 'for', 'with'}
        answer_tokens = answer_tokens - stopwords
        context_tokens = context_tokens - stopwords
        
        if not answer_tokens:
            return 0.0
        
        overlap = len(answer_tokens & context_tokens)
        return overlap / len(answer_tokens)
    
    def evaluate_dataset(
        self,
        dataset: BenchmarkDataset,
        predictions: List[str],
        retrieved_docs_list: Optional[List[List[str]]] = None
    ) -> TaskEvaluationResult:
        """Evaluate entire dataset for retrieval QA."""
        self.status = TaskStatus.RUNNING
        
        per_sample_results = []
        for i, (sample, pred) in enumerate(zip(dataset.samples, predictions)):
            retrieved_docs = retrieved_docs_list[i] if retrieved_docs_list else None
            result = self.evaluate_sample(sample, pred, retrieved_docs)
            per_sample_results.append(result)
        
        aggregate_metrics = self._aggregate_metrics(per_sample_results)
        
        self.status = TaskStatus.COMPLETED
        
        return TaskEvaluationResult(
            task_name=self.name,
            task_type=self.task_type,
            num_samples=len(dataset.samples),
            aggregate_metrics=aggregate_metrics,
            per_sample_results=per_sample_results,
            metadata={'dataset_name': dataset.name}
        )
    
    def _aggregate_metrics(self, results: List[TaskResult]) -> Dict[str, float]:
        """Aggregate retrieval QA metrics."""
        if not results:
            return {}
        
        # Collect all metric names
        all_metrics = set()
        for r in results:
            all_metrics.update(r.metrics.keys())
        
        aggregated = {}
        for metric in all_metrics:
            values = [r.metrics.get(metric, 0.0) for r in results if metric in r.metrics]
            if values:
                aggregated[f"{metric}_mean"] = float(np.mean(values))
                aggregated[f"{metric}_std"] = float(np.std(values))
        
        return aggregated


# Task factory
def create_task(task_type: TaskType, **kwargs) -> BaseTask:
    """Factory function to create tasks."""
    task_classes = {
        TaskType.CLINICAL_REASONING: ClinicalReasoningTask,
        TaskType.DIAGNOSTIC: DiagnosticTask,
        TaskType.MULTI_CHOICE: MultiChoiceTask,
        TaskType.RETRIEVAL_QA: RetrievalQATask
    }
    
    task_class = task_classes.get(task_type)
    if task_class is None:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return task_class(**kwargs)


if __name__ == "__main__":
    # Example usage
    from .datasets import BenchmarkSample
    
    # Test Clinical Reasoning Task
    task = ClinicalReasoningTask()
    
    sample = BenchmarkSample(
        id="test_001",
        input_text="A 45-year-old male presents with chest pain, shortness of breath, and sweating.",
        expected_output="Possible acute coronary syndrome. DDX includes MI, unstable angina. Immediate ECG, cardiac enzymes recommended.",
        task_type=TaskType.CLINICAL_REASONING,
        metadata={'county': 'Nairobi'}
    )
    
    prediction = "The patient may have a cardiac event. Recommend ECG and troponin levels."
    
    result = task.evaluate_sample(sample, prediction)
    print(f"Sample {result.sample_id}")
    print(f"Metrics: {result.metrics}")
