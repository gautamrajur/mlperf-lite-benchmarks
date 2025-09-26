"""SST-2 dataset loader for MLPerf Lite Benchmarks."""

import os
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

from mlperf_lite.datasets.base import BaseDataset


class SST2Dataset(BaseDataset):
    """Stanford Sentiment Treebank v2 (SST-2) dataset implementation."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        model_name: str = "bert-base-uncased",
        max_length: int = 128,
        **kwargs: Any
    ) -> None:
        """Initialize SST-2 dataset.
        
        Args:
            data_dir: Directory to store/load SST-2 data
            batch_size: Batch size for data loading
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            model_name: HuggingFace model name for tokenizer
            max_length: Maximum sequence length for tokenization
            **kwargs: Additional parameters
        """
        self.model_name = model_name
        self.max_length = max_length
        super().__init__(data_dir, batch_size, num_workers, pin_memory, **kwargs)
    
    def _load_tokenizer(self) -> None:
        """Load the tokenizer for the specified model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def _tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize examples for BERT.
        
        Args:
            examples: Batch of examples with 'sentence' and 'label' keys
            
        Returns:
            Tokenized examples
        """
        return self.tokenizer(
            examples["sentence"],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def _load_datasets(self) -> None:
        """Load SST-2 train, validation, and test datasets."""
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load tokenizer
        self._load_tokenizer()
        
        # Load SST-2 dataset from HuggingFace
        dataset = load_dataset("glue", "sst2")
        
        # Tokenize the datasets
        train_dataset = dataset["train"].map(
            self._tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        val_dataset = dataset["validation"].map(
            self._tokenize_function,
            batched=True,
            remove_columns=dataset["validation"].column_names
        )
        
        # Convert to PyTorch datasets
        self.train_dataset = SST2PyTorchDataset(train_dataset)
        self.val_dataset = SST2PyTorchDataset(val_dataset)
        self.test_dataset = SST2PyTorchDataset(val_dataset)  # Use val as test for SST-2
    
    def get_train_loader(self) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
    
    def get_val_loader(self) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def get_test_loader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def get_num_classes(self) -> int:
        """Get number of classes in SST-2."""
        return 2
    
    def get_input_shape(self) -> Tuple[int, ...]:
        """Get input shape for SST-2 (sequence length)."""
        return (self.max_length,)
    
    def get_class_names(self) -> list[str]:
        """Get class names for SST-2."""
        return ["negative", "positive"]
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get SST-2 specific dataset information."""
        info = super().get_dataset_info()
        info.update({
            "dataset_type": "SST-2",
            "model_name": self.model_name,
            "max_length": self.max_length,
            "vocab_size": self.tokenizer.vocab_size,
            "task_type": "sentiment_classification",
        })
        return info


class SST2PyTorchDataset(Dataset):
    """PyTorch Dataset wrapper for SST-2 HuggingFace dataset."""
    
    def __init__(self, hf_dataset) -> None:
        """Initialize with HuggingFace dataset.
        
        Args:
            hf_dataset: HuggingFace dataset object
        """
        self.hf_dataset = hf_dataset
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.hf_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item at index.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        item = self.hf_dataset[idx]
        return {
            "input_ids": item["input_ids"].squeeze(0),
            "attention_mask": item["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }
