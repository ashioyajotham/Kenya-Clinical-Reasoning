# Kenya Clinical Reasoning Benchmark Suite
# Inspired by MedQA and African Medical QA benchmarks

"""
Kenya Clinical Reasoning Benchmark Suite

A comprehensive evaluation framework for clinical reasoning models,
designed specifically for the Kenya Clinical Reasoning Challenge.

Evaluation Components:
1. Diagnostic Accuracy - DDX prediction evaluation
2. Clinical Reasoning Quality - Free-text response evaluation  
3. Retrieval-Based QA - Knowledge retrieval and answer generation
4. Multi-Choice Clinical QA - Structured clinical scenarios

Metrics:
- ROUGE (1, 2, L) for text similarity
- Semantic Similarity (sentence-transformers)
- Exact Match for structured outputs
- Clinical F1 for diagnosis accuracy
- Retrieval metrics (MRR, Recall@K)

Example Usage:
    from Benchmarking import BenchmarkRunner, BenchmarkConfig, ModelConfig
    
    runner = BenchmarkRunner(config=BenchmarkConfig(), data_dir=".")
    runner.setup()
    
    model_config = ModelConfig(name="my_model", model_fn=my_inference_fn)
    result = runner.evaluate_model(model_config)
    print(runner.generate_report())
"""

__version__ = "1.0.0"
__author__ = "Kenya Clinical Reasoning Team"

# Core components
from .benchmark_runner import (
    BenchmarkRunner,
    BenchmarkConfig,
    BenchmarkResult,
    ModelConfig,
    run_kenya_clinical_benchmark
)

from .metrics import ClinicalMetrics, BootstrapEvaluator

from .datasets import (
    TaskType,
    BenchmarkSample,
    BenchmarkDataset,
    KenyaClinicalDataLoader
)

from .retrieval import (
    ClinicalRetriever,
    ClinicalKnowledgeBase,
    Document,
    RetrievalResult
)

from .tasks import (
    BaseTask,
    ClinicalReasoningTask,
    DiagnosticTask,
    MultiChoiceTask,
    RetrievalQATask,
    TaskResult,
    TaskEvaluationResult,
    TaskStatus,
    create_task
)

__all__ = [
    # Runner
    "BenchmarkRunner",
    "BenchmarkConfig",
    "BenchmarkResult",
    "ModelConfig",
    "run_kenya_clinical_benchmark",
    
    # Metrics
    "ClinicalMetrics",
    "BootstrapEvaluator",
    
    # Datasets
    "TaskType",
    "BenchmarkSample",
    "BenchmarkDataset",
    "KenyaClinicalDataLoader",
    
    # Retrieval
    "ClinicalRetriever",
    "ClinicalKnowledgeBase",
    "Document",
    "RetrievalResult",
    
    # Tasks
    "BaseTask",
    "ClinicalReasoningTask",
    "DiagnosticTask",
    "MultiChoiceTask",
    "RetrievalQATask",
    "TaskResult",
    "TaskEvaluationResult",
    "TaskStatus",
    "create_task",
]
