"""
Benchmark Runner Module

Main orchestration class that coordinates:
- Dataset loading and preprocessing
- Model inference
- Task evaluation
- Results aggregation and reporting
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import warnings

from .datasets import (
    TaskType, BenchmarkSample, BenchmarkDataset,
    KenyaClinicalDataLoader
)
from .metrics import ClinicalMetrics, BootstrapEvaluator
from .tasks import (
    BaseTask, ClinicalReasoningTask, DiagnosticTask,
    MultiChoiceTask, RetrievalQATask, TaskEvaluationResult,
    create_task
)
from .retrieval import ClinicalRetriever, ClinicalKnowledgeBase


@dataclass
class ModelConfig:
    """Configuration for a model to be evaluated."""
    name: str
    model_fn: Callable[[str], str]  # Function that takes prompt, returns response
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark run."""
    name: str = "kenya_clinical_benchmark"
    version: str = "1.0.0"
    description: str = "Kenya Clinical Reasoning Benchmark Suite"
    
    # Task selection
    tasks: List[str] = field(default_factory=lambda: [
        "clinical_reasoning", "diagnostic", "multi_choice", "retrieval_qa"
    ])
    
    # Evaluation settings
    compute_ci: bool = True
    ci_samples: int = 1000
    ci_confidence: float = 0.95
    
    # Output settings
    output_dir: str = "benchmark_results"
    save_predictions: bool = True
    save_per_sample: bool = True
    
    # Data settings
    train_data_path: str = "train.csv"
    test_data_path: str = "test.csv"
    val_split: float = 0.15
    random_seed: int = 42


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a model."""
    model_name: str
    config: BenchmarkConfig
    task_results: Dict[str, TaskEvaluationResult]
    aggregate_score: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'config': asdict(self.config),
            'aggregate_score': self.aggregate_score,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'task_results': {
                task_name: {
                    'task_name': result.task_name,
                    'task_type': result.task_type.value,
                    'num_samples': result.num_samples,
                    'aggregate_metrics': result.aggregate_metrics,
                    'metadata': result.metadata
                }
                for task_name, result in self.task_results.items()
            }
        }


class BenchmarkRunner:
    """
    Main benchmark orchestration class.
    
    Coordinates dataset loading, model evaluation, and result reporting.
    """
    
    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        data_dir: Optional[str] = None
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            config: Benchmark configuration
            data_dir: Directory containing data files
        """
        self.config = config or BenchmarkConfig()
        self.data_dir = Path(data_dir) if data_dir else Path(".")
        
        self.metrics = ClinicalMetrics()
        self.bootstrap = BootstrapEvaluator() if self.config.compute_ci else None
        
        # Components
        self.data_loader: Optional[KenyaClinicalDataLoader] = None
        self.datasets: Dict[str, BenchmarkDataset] = {}
        self.tasks: Dict[str, BaseTask] = {}
        self.retriever: Optional[ClinicalRetriever] = None
        self.knowledge_base: Optional[ClinicalKnowledgeBase] = None
        
        # Results
        self.results: Dict[str, BenchmarkResult] = {}
        
    def setup(self):
        """Initialize all components."""
        print("Setting up benchmark...")
        
        # Load data
        train_path = self.data_dir / self.config.train_data_path
        self.data_loader = KenyaClinicalDataLoader(
            train_path=str(train_path),
            val_split=self.config.val_split,
            random_seed=self.config.random_seed
        )
        
        # Create datasets for each task
        self._create_datasets()
        
        # Create task evaluators
        self._create_tasks()
        
        # Setup retriever if needed
        if "retrieval_qa" in self.config.tasks:
            self._setup_retrieval()
        
        print(f"Setup complete. {len(self.datasets)} datasets, {len(self.tasks)} tasks.")
    
    def _create_datasets(self):
        """Create benchmark datasets for each task type."""
        print("Creating benchmark datasets...")
        
        task_creators = {
            "clinical_reasoning": self.data_loader.create_clinical_reasoning_dataset,
            "diagnostic": self.data_loader.create_diagnostic_dataset,
            "multi_choice": self.data_loader.create_multi_choice_dataset,
            "retrieval_qa": self.data_loader.create_retrieval_qa_dataset
        }
        
        for task_name in self.config.tasks:
            if task_name in task_creators:
                try:
                    self.datasets[task_name] = task_creators[task_name]()
                    print(f"  Created {task_name} dataset: {len(self.datasets[task_name].samples)} samples")
                except Exception as e:
                    warnings.warn(f"Failed to create {task_name} dataset: {e}")
    
    def _create_tasks(self):
        """Create task evaluators."""
        print("Creating task evaluators...")
        
        task_mapping = {
            "clinical_reasoning": TaskType.CLINICAL_REASONING,
            "diagnostic": TaskType.DIAGNOSTIC,
            "multi_choice": TaskType.MULTI_CHOICE,
            "retrieval_qa": TaskType.RETRIEVAL_QA
        }
        
        for task_name in self.config.tasks:
            if task_name in task_mapping and task_name in self.datasets:
                self.tasks[task_name] = create_task(
                    task_mapping[task_name],
                    name=task_name,
                    metrics=self.metrics
                )
                print(f"  Created {task_name} task evaluator")
    
    def _setup_retrieval(self):
        """Setup retrieval components."""
        print("Setting up retrieval system...")
        
        self.retriever = ClinicalRetriever()
        self.knowledge_base = ClinicalKnowledgeBase()
        
        # Build knowledge base from training data
        if self.data_loader is not None:
            try:
                self.knowledge_base.build_from_responses(
                    self.data_loader.train_df,
                    response_columns=['Clinician', 'GPT4.0', 'LLAMA', 'GEMINI']
                )
                print(f"  Knowledge base built with {len(self.knowledge_base.retriever.documents)} documents")
            except Exception as e:
                warnings.warn(f"Failed to build knowledge base: {e}")
    
    def evaluate_model(
        self,
        model_config: ModelConfig,
        tasks: Optional[List[str]] = None
    ) -> BenchmarkResult:
        """
        Evaluate a model on the benchmark.
        
        Args:
            model_config: Model configuration with inference function
            tasks: Specific tasks to run (default: all configured tasks)
            
        Returns:
            BenchmarkResult with all evaluation results
        """
        tasks_to_run = tasks or self.config.tasks
        task_results = {}
        
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_config.name}")
        print(f"{'='*60}")
        
        for task_name in tasks_to_run:
            if task_name not in self.tasks or task_name not in self.datasets:
                warnings.warn(f"Skipping {task_name}: not configured")
                continue
            
            print(f"\n--- Task: {task_name} ---")
            
            task = self.tasks[task_name]
            dataset = self.datasets[task_name]
            
            # Generate predictions
            predictions = self._generate_predictions(
                model_config.model_fn,
                task,
                dataset
            )
            
            # Evaluate
            result = task.evaluate_dataset(dataset, predictions)
            task_results[task_name] = result
            
            # Print summary
            self._print_task_summary(result)
        
        # Compute aggregate score
        aggregate_score = self._compute_aggregate_score(task_results)
        
        # Create result
        benchmark_result = BenchmarkResult(
            model_name=model_config.name,
            config=self.config,
            task_results=task_results,
            aggregate_score=aggregate_score,
            timestamp=datetime.now().isoformat(),
            metadata=model_config.metadata
        )
        
        # Store result
        self.results[model_config.name] = benchmark_result
        
        # Save results
        if self.config.output_dir:
            self._save_results(benchmark_result)
        
        return benchmark_result
    
    def _generate_predictions(
        self,
        model_fn: Callable[[str], str],
        task: BaseTask,
        dataset: BenchmarkDataset
    ) -> List[str]:
        """Generate predictions for all samples in a dataset."""
        predictions = []
        
        for i, sample in enumerate(dataset.samples):
            # Format prompt
            if hasattr(task, 'format_prompt'):
                prompt = task.format_prompt(sample)
            else:
                prompt = task.get_prompt_template().format(input=sample.input_text)
            
            # Generate prediction
            try:
                prediction = model_fn(prompt)
            except Exception as e:
                warnings.warn(f"Error generating prediction for sample {sample.id}: {e}")
                prediction = ""
            
            predictions.append(prediction)
            
            # Progress
            if (i + 1) % 10 == 0 or (i + 1) == len(dataset.samples):
                print(f"  Generated {i + 1}/{len(dataset.samples)} predictions", end='\r')
        
        print()  # New line after progress
        return predictions
    
    def _print_task_summary(self, result: TaskEvaluationResult):
        """Print summary of task results."""
        print(f"  Samples: {result.num_samples}")
        print(f"  Key metrics:")
        
        # Show most relevant metrics
        priority_metrics = [
            'accuracy', 'rouge_1_mean', 'rouge_l_mean',
            'semantic_similarity_mean', 'exact_match_mean', 'mrr'
        ]
        
        for metric in priority_metrics:
            if metric in result.aggregate_metrics:
                value = result.aggregate_metrics[metric]
                print(f"    {metric}: {value:.4f}")
    
    def _compute_aggregate_score(
        self,
        task_results: Dict[str, TaskEvaluationResult]
    ) -> float:
        """
        Compute weighted aggregate score across all tasks.
        
        Weights tasks based on their importance for clinical reasoning.
        """
        if not task_results:
            return 0.0
        
        # Task weights
        weights = {
            "clinical_reasoning": 0.35,
            "diagnostic": 0.30,
            "multi_choice": 0.15,
            "retrieval_qa": 0.20
        }
        
        # Primary metrics per task
        primary_metrics = {
            "clinical_reasoning": "semantic_similarity_mean",
            "diagnostic": "top3_accuracy_mean",
            "multi_choice": "accuracy",
            "retrieval_qa": "semantic_similarity_mean"
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for task_name, result in task_results.items():
            weight = weights.get(task_name, 0.25)
            metric_name = primary_metrics.get(task_name, "accuracy")
            
            # Get metric value
            metric_value = result.aggregate_metrics.get(metric_name, 0.0)
            
            # Fallback to any available metric
            if metric_value == 0.0 and result.aggregate_metrics:
                metric_value = list(result.aggregate_metrics.values())[0]
            
            weighted_sum += weight * metric_value
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _save_results(self, result: BenchmarkResult):
        """Save benchmark results to disk."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Main results file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.model_name}_{timestamp}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        
        # Save per-sample results if configured
        if self.config.save_per_sample:
            for task_name, task_result in result.task_results.items():
                sample_filepath = output_dir / f"{result.model_name}_{task_name}_samples_{timestamp}.json"
                
                samples_data = [
                    {
                        'sample_id': r.sample_id,
                        'prediction': r.prediction,
                        'ground_truth': r.ground_truth,
                        'metrics': r.metrics
                    }
                    for r in task_result.per_sample_results
                ]
                
                with open(sample_filepath, 'w') as f:
                    json.dump(samples_data, f, indent=2)
    
    def compare_models(self, model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare multiple evaluated models.
        
        Args:
            model_names: Names of models to compare (default: all evaluated)
            
        Returns:
            Comparison dictionary with rankings and metrics
        """
        models = model_names or list(self.results.keys())
        
        if len(models) < 2:
            print("Need at least 2 models to compare")
            return {}
        
        comparison = {
            'models': models,
            'aggregate_scores': {},
            'task_scores': {},
            'ranking': []
        }
        
        # Collect scores
        for model in models:
            if model not in self.results:
                continue
            
            result = self.results[model]
            comparison['aggregate_scores'][model] = result.aggregate_score
            comparison['task_scores'][model] = {
                task_name: task_result.aggregate_metrics
                for task_name, task_result in result.task_results.items()
            }
        
        # Rank by aggregate score
        comparison['ranking'] = sorted(
            models,
            key=lambda m: comparison['aggregate_scores'].get(m, 0),
            reverse=True
        )
        
        return comparison
    
    def generate_report(self) -> str:
        """Generate a markdown report of all benchmark results."""
        lines = [
            f"# {self.config.name} Benchmark Report",
            f"Version: {self.config.version}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            ""
        ]
        
        if not self.results:
            lines.append("No models evaluated yet.")
            return "\n".join(lines)
        
        # Summary table
        lines.append("| Model | Aggregate Score | Tasks Evaluated |")
        lines.append("|-------|-----------------|-----------------|")
        
        for model_name, result in sorted(
            self.results.items(),
            key=lambda x: x[1].aggregate_score,
            reverse=True
        ):
            tasks = len(result.task_results)
            lines.append(f"| {model_name} | {result.aggregate_score:.4f} | {tasks} |")
        
        lines.extend(["", "## Detailed Results", ""])
        
        for model_name, result in self.results.items():
            lines.extend([
                f"### {model_name}",
                "",
                f"**Aggregate Score:** {result.aggregate_score:.4f}",
                f"**Timestamp:** {result.timestamp}",
                ""
            ])
            
            for task_name, task_result in result.task_results.items():
                lines.extend([
                    f"#### {task_name}",
                    f"- Samples: {task_result.num_samples}",
                    "- Metrics:"
                ])
                
                for metric, value in sorted(task_result.aggregate_metrics.items()):
                    lines.append(f"  - {metric}: {value:.4f}")
                
                lines.append("")
        
        return "\n".join(lines)


# Convenience function to run benchmark
def run_kenya_clinical_benchmark(
    model_fn: Callable[[str], str],
    model_name: str,
    data_dir: str = ".",
    tasks: Optional[List[str]] = None,
    **config_kwargs
) -> BenchmarkResult:
    """
    Run the Kenya Clinical benchmark on a model.
    
    Args:
        model_fn: Function that takes prompt and returns response
        model_name: Name for the model
        data_dir: Directory containing data files
        tasks: List of tasks to run
        **config_kwargs: Additional config parameters
        
    Returns:
        BenchmarkResult with evaluation results
    """
    config = BenchmarkConfig(**config_kwargs)
    if tasks:
        config.tasks = tasks
    
    runner = BenchmarkRunner(config=config, data_dir=data_dir)
    runner.setup()
    
    model_config = ModelConfig(name=model_name, model_fn=model_fn)
    result = runner.evaluate_model(model_config)
    
    return result


if __name__ == "__main__":
    # Example usage
    print("Kenya Clinical Benchmark Runner")
    print("================================")
    
    # Example dummy model
    def dummy_model(prompt: str) -> str:
        return "This is a test response for clinical evaluation."
    
    # Run benchmark
    config = BenchmarkConfig(
        name="test_benchmark",
        tasks=["clinical_reasoning"],
        output_dir="./benchmark_results"
    )
    
    runner = BenchmarkRunner(config=config, data_dir=".")
    
    print("\nTo run the benchmark:")
    print("  runner.setup()")
    print("  model_config = ModelConfig(name='my_model', model_fn=my_inference_function)")
    print("  result = runner.evaluate_model(model_config)")
    print("  print(runner.generate_report())")
