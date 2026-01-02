"""
Benchmark Dataset Module

Handles loading, preprocessing, and splitting of the Kenya Clinical
Reasoning Challenge data for benchmarking purposes.

Dataset Types:
1. Clinical Reasoning (free-text generation)
2. Diagnostic Accuracy (DDX prediction)
3. Multi-Choice QA (generated from clinical scenarios)
4. Retrieval QA (knowledge-grounded QA)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from enum import Enum
import random


class TaskType(Enum):
    """Types of benchmark tasks."""
    CLINICAL_REASONING = "clinical_reasoning"
    DIAGNOSTIC = "diagnostic"
    MULTI_CHOICE = "multi_choice"
    RETRIEVAL_QA = "retrieval_qa"


@dataclass
class BenchmarkSample:
    """Single benchmark sample."""
    id: str
    input_text: str  # The prompt/question
    expected_output: str  # The reference answer
    task_type: TaskType
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # For multi-choice tasks
    choices: Optional[List[str]] = None
    correct_answer: Optional[int] = None
    
    # For retrieval tasks
    context_docs: Optional[List[str]] = None
    relevant_doc_ids: Optional[List[str]] = None
    
    # Aliases for backward compatibility
    @property
    def prompt(self) -> str:
        return self.input_text
    
    @property
    def reference(self) -> str:
        return self.expected_output


@dataclass
class BenchmarkDataset:
    """
    Container for benchmark samples with train/val/test splits.
    """
    name: str
    task_type: TaskType
    train: List[BenchmarkSample]
    val: List[BenchmarkSample]
    test: List[BenchmarkSample]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def samples(self) -> List[BenchmarkSample]:
        """Get all samples (train + val + test)."""
        return self.train + self.val + self.test
    
    def __len__(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)
    
    def get_split(self, split: str) -> List[BenchmarkSample]:
        """Get samples for a specific split."""
        if split == "train":
            return self.train
        elif split == "val" or split == "validation":
            return self.val
        elif split == "test":
            return self.test
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "task_type": self.task_type.value,
            "train_size": len(self.train),
            "val_size": len(self.val),
            "test_size": len(self.test),
            "metadata": self.metadata
        }


class KenyaClinicalDataLoader:
    """
    Data loader for Kenya Clinical Reasoning Challenge datasets.
    
    Supports:
    - Loading train/test CSV files
    - Creating benchmark datasets for different task types
    - Preprocessing and cleaning clinical text
    """
    
    def __init__(
        self,
        data_dir: str = ".",
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        val_split: float = 0.15,
        random_seed: int = 42
    ):
        """
        Args:
            data_dir: Directory containing train.csv, test.csv, etc.
            train_path: Direct path to train file (overrides data_dir)
            test_path: Direct path to test file (overrides data_dir)
            val_split: Fraction of training data to use for validation
            random_seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.train_path = Path(train_path) if train_path else None
        self.test_path = Path(test_path) if test_path else None
        self.val_split = val_split
        self.random_seed = random_seed
        
        self._train_df = None
        self._test_df = None
        self._train_raw_df = None
        self._test_raw_df = None
    
    @property
    def train_df(self) -> pd.DataFrame:
        """Lazy load training data."""
        if self._train_df is None:
            if self.train_path and self.train_path.exists():
                train_file = self.train_path
            else:
                train_file = self.data_dir / "train.csv"
            
            if train_file.exists():
                self._train_df = pd.read_csv(train_file)
            else:
                raise FileNotFoundError(f"Train file not found: {train_file}")
        return self._train_df
    
    @property
    def test_df(self) -> pd.DataFrame:
        """Lazy load test data."""
        if self._test_df is None:
            if self.test_path and self.test_path.exists():
                test_file = self.test_path
            else:
                test_file = self.data_dir / "test.csv"
            
            if test_file.exists():
                self._test_df = pd.read_csv(test_file)
            else:
                raise FileNotFoundError(f"Test file not found: {test_file}")
        return self._test_df
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Clean and preprocess clinical text."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        text = text.lower().strip()
        text = re.sub(r'[^\w\s\']', '', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @staticmethod
    def extract_diagnosis(text: str) -> str:
        """Extract diagnosis from clinical response."""
        if not text:
            return ""
        
        text_lower = text.lower()
        
        # Patterns to match diagnosis sections
        patterns = [
            r'diagnosis[:\s]+([^.]+)',
            r'ddx[:\s]+([^.]+)',
            r'differential[:\s]+([^.]+)',
            r'impression[:\s]+([^.]+)',
            r'assessment[:\s]+([^.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def create_clinical_reasoning_dataset(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> BenchmarkDataset:
        """
        Create clinical reasoning benchmark dataset.
        
        Task: Given a clinical prompt, generate a comprehensive clinical response.
        """
        np.random.seed(seed)
        
        samples = []
        for idx, row in self.train_df.iterrows():
            sample = BenchmarkSample(
                id=str(row.get('Master_Index', f'train_{idx}')),
                input_text=str(row.get('Prompt', '')),
                expected_output=str(row.get('Clinician', '')),
                task_type=TaskType.CLINICAL_REASONING,
                metadata={
                    'county': row.get('County', ''),
                    'health_level': row.get('Health level', ''),
                    'years_experience': row.get('Years of Experience', 0),
                    'nursing_competency': row.get('Nursing Competency', ''),
                    'clinical_panel': row.get('Clinical Panel', ''),
                    'ddx_snomed': row.get('DDX SNOMED', ''),
                    # Include other model responses for comparison
                    'gpt4_response': row.get('GPT4.0', ''),
                    'llama_response': row.get('LLAMA', ''),
                    'gemini_response': row.get('GEMINI', '')
                }
            )
            samples.append(sample)
        
        # Shuffle and split
        np.random.shuffle(samples)
        n_train = int(len(samples) * train_ratio)
        n_val = int(len(samples) * val_ratio)
        
        train_samples = samples[:n_train]
        val_samples = samples[n_train:n_train + n_val]
        test_samples = samples[n_train + n_val:]
        
        return BenchmarkDataset(
            name="kenya_clinical_reasoning",
            task_type=TaskType.CLINICAL_REASONING,
            train=train_samples,
            val=val_samples,
            test=test_samples,
            metadata={
                'source': 'Kenya Clinical Reasoning Challenge',
                'task_description': 'Generate comprehensive clinical responses to patient scenarios',
                'metrics': ['rouge', 'semantic_similarity', 'completeness']
            }
        )
    
    def create_diagnostic_dataset(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> BenchmarkDataset:
        """
        Create diagnostic accuracy benchmark dataset.
        
        Task: Given a clinical prompt, predict the correct diagnosis.
        """
        np.random.seed(seed)
        
        samples = []
        for idx, row in self.train_df.iterrows():
            # Extract diagnosis from reference
            clinician_text = str(row.get('Clinician', ''))
            diagnosis = self.extract_diagnosis(clinician_text)
            snomed_code = row.get('DDX SNOMED', '')
            
            if not diagnosis and not snomed_code:
                continue  # Skip if no diagnosis available
            
            sample = BenchmarkSample(
                id=str(row.get('Master_Index', f'diag_{idx}')),
                input_text=str(row.get('Prompt', '')),
                expected_output=diagnosis if diagnosis else str(snomed_code),
                task_type=TaskType.DIAGNOSTIC,
                metadata={
                    'snomed_code': str(snomed_code),
                    'full_response': clinician_text,
                    'clinical_panel': row.get('Clinical Panel', ''),
                    'county': row.get('County', ''),
                    'ddx_snomed': str(snomed_code)
                }
            )
            samples.append(sample)
        
        # Shuffle and split
        np.random.shuffle(samples)
        n_train = int(len(samples) * train_ratio)
        n_val = int(len(samples) * val_ratio)
        
        return BenchmarkDataset(
            name="kenya_diagnostic",
            task_type=TaskType.DIAGNOSTIC,
            train=samples[:n_train],
            val=samples[n_train:n_train + n_val],
            test=samples[n_train + n_val:],
            metadata={
                'source': 'Kenya Clinical Reasoning Challenge',
                'task_description': 'Predict diagnosis from clinical presentation',
                'metrics': ['ddx_accuracy', 'snomed_match']
            }
        )
    
    def create_multi_choice_dataset(
        self,
        n_distractors: int = 3,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> BenchmarkDataset:
        """
        Create multi-choice QA benchmark dataset (MedQA-style).
        
        Task: Given a clinical scenario, select the correct answer from choices.
        """
        np.random.seed(seed)
        random.seed(seed)
        
        # First, collect all unique diagnoses for distractor generation
        all_diagnoses = []
        for idx, row in self.train_df.iterrows():
            clinician_text = str(row.get('Clinician', ''))
            diagnosis = self.extract_diagnosis(clinician_text)
            if diagnosis:
                all_diagnoses.append(diagnosis)
        
        unique_diagnoses = list(set(all_diagnoses))
        
        samples = []
        for idx, row in self.train_df.iterrows():
            clinician_text = str(row.get('Clinician', ''))
            correct_diagnosis = self.extract_diagnosis(clinician_text)
            
            if not correct_diagnosis:
                continue
            
            # Generate distractors (other diagnoses)
            distractors = [d for d in unique_diagnoses if d != correct_diagnosis]
            if len(distractors) < n_distractors:
                continue
            
            selected_distractors = random.sample(distractors, n_distractors)
            
            # Create choices with randomized position
            choices = selected_distractors + [correct_diagnosis]
            random.shuffle(choices)
            correct_idx = choices.index(correct_diagnosis)
            
            # Create question prompt
            question = f"Based on the following clinical presentation, what is the most likely diagnosis?\n\n{row.get('Prompt', '')}"
            
            sample = BenchmarkSample(
                id=str(row.get('Master_Index', f'mc_{idx}')),
                input_text=question,
                expected_output=correct_diagnosis,
                task_type=TaskType.MULTI_CHOICE,
                choices=choices,
                correct_answer=correct_idx,
                metadata={
                    'clinical_panel': row.get('Clinical Panel', ''),
                    'county': row.get('County', ''),
                    'original_response': clinician_text,
                    'correct_answer': chr(65 + correct_idx)  # A, B, C, D...
                }
            )
            samples.append(sample)
        
        # Shuffle and split
        np.random.shuffle(samples)
        n_train = int(len(samples) * train_ratio)
        n_val = int(len(samples) * val_ratio)
        
        return BenchmarkDataset(
            name="kenya_multi_choice",
            task_type=TaskType.MULTI_CHOICE,
            train=samples[:n_train],
            val=samples[n_train:n_train + n_val],
            test=samples[n_train + n_val:],
            metadata={
                'source': 'Kenya Clinical Reasoning Challenge',
                'task_description': 'Multiple choice diagnostic question answering',
                'n_choices': n_distractors + 1,
                'metrics': ['accuracy', 'f1']
            }
        )
    
    def create_retrieval_qa_dataset(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> BenchmarkDataset:
        """
        Create retrieval-augmented QA benchmark dataset.
        
        Task: Given a clinical question, retrieve relevant context and generate answer.
        """
        np.random.seed(seed)
        
        # Build knowledge base from training responses
        knowledge_base = []
        for idx, row in self.train_df.iterrows():
            for response_col in ['Clinician', 'GPT4.0', 'LLAMA', 'GEMINI']:
                response = str(row.get(response_col, ''))
                if response and len(response) > 50:
                    knowledge_base.append({
                        'id': f'{response_col}_{idx}',
                        'text': response,
                        'source': response_col,
                        'clinical_panel': row.get('Clinical Panel', ''),
                        'prompt_idx': idx
                    })
        
        samples = []
        for idx, row in self.train_df.iterrows():
            prompt = str(row.get('Prompt', ''))
            clinician_response = str(row.get('Clinician', ''))
            
            if not prompt or not clinician_response:
                continue
            
            # Find relevant documents (from same clinical panel or similar cases)
            clinical_panel = row.get('Clinical Panel', '')
            relevant_docs = [
                doc['id'] for doc in knowledge_base
                if doc['clinical_panel'] == clinical_panel and doc['prompt_idx'] != idx
            ][:5]  # Limit to 5 relevant docs
            
            sample = BenchmarkSample(
                id=str(row.get('Master_Index', f'ret_{idx}')),
                input_text=prompt,
                expected_output=clinician_response,
                task_type=TaskType.RETRIEVAL_QA,
                relevant_doc_ids=relevant_docs,
                metadata={
                    'clinical_panel': clinical_panel,
                    'county': row.get('County', ''),
                    'relevant_docs': relevant_docs
                }
            )
            samples.append(sample)
        
        # Shuffle and split
        np.random.shuffle(samples)
        n_train = int(len(samples) * train_ratio)
        n_val = int(len(samples) * val_ratio)
        
        dataset = BenchmarkDataset(
            name="kenya_retrieval_qa",
            task_type=TaskType.RETRIEVAL_QA,
            train=samples[:n_train],
            val=samples[n_train:n_train + n_val],
            test=samples[n_train + n_val:],
            metadata={
                'source': 'Kenya Clinical Reasoning Challenge',
                'task_description': 'Retrieval-augmented clinical question answering',
                'knowledge_base_size': len(knowledge_base),
                'metrics': ['mrr', 'recall@k', 'generation_quality']
            }
        )
        
        # Store knowledge base for retrieval
        dataset.metadata['knowledge_base'] = knowledge_base
        
        return dataset
    
    def export_to_jsonl(self, dataset: BenchmarkDataset, output_dir: str):
        """Export dataset to JSONL format for compatibility."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name in ['train', 'val', 'test']:
            samples = dataset.get_split(split_name)
            output_file = output_path / f"{dataset.name}_{split_name}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    record = {
                        'id': sample.id,
                        'prompt': sample.prompt,
                        'reference': sample.reference,
                        'task_type': sample.task_type.value,
                        'metadata': sample.metadata
                    }
                    
                    if sample.choices:
                        record['choices'] = sample.choices
                        record['correct_answer'] = sample.correct_answer
                    
                    if sample.relevant_doc_ids:
                        record['relevant_doc_ids'] = sample.relevant_doc_ids
                    
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        # Save metadata
        meta_file = output_path / f"{dataset.name}_metadata.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(dataset.to_dict(), f, indent=2)


class DatasetIterator:
    """Batch iterator for benchmark datasets."""
    
    def __init__(
        self,
        dataset: BenchmarkDataset,
        split: str = "train",
        batch_size: int = 32,
        shuffle: bool = True
    ):
        self.samples = dataset.get_split(split)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._indices = None
        self._position = 0
    
    def __len__(self) -> int:
        return (len(self.samples) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[List[BenchmarkSample]]:
        self._indices = list(range(len(self.samples)))
        if self.shuffle:
            np.random.shuffle(self._indices)
        self._position = 0
        return self
    
    def __next__(self) -> List[BenchmarkSample]:
        if self._position >= len(self._indices):
            raise StopIteration
        
        batch_indices = self._indices[self._position:self._position + self.batch_size]
        self._position += self.batch_size
        
        return [self.samples[i] for i in batch_indices]


if __name__ == "__main__":
    # Example usage
    loader = KenyaClinicalDataLoader(".")
    
    try:
        # Create different benchmark datasets
        clinical_dataset = loader.create_clinical_reasoning_dataset()
        diagnostic_dataset = loader.create_diagnostic_dataset()
        mc_dataset = loader.create_multi_choice_dataset()
        retrieval_dataset = loader.create_retrieval_qa_dataset()
        
        print("=== Dataset Statistics ===")
        for ds in [clinical_dataset, diagnostic_dataset, mc_dataset, retrieval_dataset]:
            print(f"\n{ds.name}:")
            print(f"  Train: {len(ds.train)}, Val: {len(ds.val)}, Test: {len(ds.test)}")
            print(f"  Task: {ds.task_type.value}")
        
        # Export to JSONL
        loader.export_to_jsonl(clinical_dataset, "benchmark_data")
        
    except FileNotFoundError as e:
        print(f"Data files not found: {e}")
