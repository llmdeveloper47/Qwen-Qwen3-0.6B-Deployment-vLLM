#!/usr/bin/env python3
"""
Local benchmarking script for vLLM inference with different configurations.
Tests various batch sizes and quantization methods to find optimal settings.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from vllm import LLM


def load_test_data(dataset_name: str = "codefactory4791/amazon_test", 
                   split: str = "test", 
                   num_samples: int = 1000,
                   balance_lengths: bool = True) -> tuple:
    """
    Load test dataset and return prompts and labels.
    Optionally balance the dataset to include short, medium, and long sequences.
    """
    print(f"Loading dataset: {dataset_name} (split={split})")
    
    dataset = load_dataset(dataset_name)
    df = dataset[split].to_pandas()
    
    # Rename columns if needed
    if 'query' in df.columns and 'label' in df.columns:
        df = df.rename(columns={'query': 'text', 'label': 'labels'})
    
    # Add text length column
    df['text_length'] = df['text'].str.len()
    
    if balance_lengths and num_samples:
        # Categorize by length (character count)
        # Short: < 200 chars, Medium: 200-500 chars, Long: > 500 chars
        df['length_category'] = 'medium'
        df.loc[df['text_length'] < 200, 'length_category'] = 'short'
        df.loc[df['text_length'] > 500, 'length_category'] = 'long'
        
        # Calculate samples per category (roughly 1/3 each)
        samples_per_category = num_samples // 3
        remainder = num_samples % 3
        
        # Sample from each category
        dfs = []
        for i, category in enumerate(['short', 'medium', 'long']):
            cat_df = df[df['length_category'] == category]
            n_samples = samples_per_category + (1 if i < remainder else 0)
            
            if len(cat_df) >= n_samples:
                sampled = cat_df.sample(n=n_samples, random_state=42)
            else:
                # If not enough samples in category, take all and adjust
                sampled = cat_df
                print(f"  Warning: Only {len(cat_df)} {category} samples available (requested {n_samples})")
            
            dfs.append(sampled)
        
        # Combine and shuffle
        df = pd.concat(dfs, ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Print distribution
        print(f"  Sampled {len(df)} instances with length distribution:")
        for category in ['short', 'medium', 'long']:
            count = len(df[df['length_category'] == category])
            avg_len = df[df['length_category'] == category]['text_length'].mean()
            print(f"    {category.capitalize()}: {count} samples (avg {avg_len:.0f} chars)")
    
    elif num_samples and num_samples < len(df):
        # Simple random sampling without balancing
        df = df.sample(n=num_samples, random_state=42)
        print(f"  Sampled {num_samples} instances for testing")
        avg_length = df['text_length'].mean()
        print(f"    Average length: {avg_length:.0f} characters")
    
    # Extract data
    prompts = df['text'].tolist()
    
    # Build label mappings
    labels_unique = sorted(df['labels'].unique())
    label2id = {lbl: idx for idx, lbl in enumerate(labels_unique)}
    id2label = {idx: lbl for idx, lbl in enumerate(labels_unique)}
    df['label_id'] = df['labels'].map(label2id)
    true_labels = df['label_id'].tolist()
    
    print(f"  Loaded {len(prompts)} prompts with {len(labels_unique)} classes")
    
    return prompts, true_labels, id2label


def initialize_model(model_id: str, quantization: str = "none") -> LLM:
    """Initialize vLLM model with specified quantization."""
    print(f"\nInitializing vLLM model...")
    print(f"  Model: {model_id}")
    print(f"  Quantization: {quantization}")
    
    start = time.perf_counter()
    
    try:
        # Handle quantization parameter
        quant_param = None if quantization == "none" else quantization
        
        llm = LLM(
            model=model_id,
            task="classify",
            max_num_seqs=32,  # Will vary in batching tests
            max_model_len=512,
            quantization=quant_param,
            dtype="auto",
            gpu_memory_utilization=0.95,
            trust_remote_code=True,
            enforce_eager=True,
        )
        
        load_time = time.perf_counter() - start
        print(f"  ✓ Model loaded in {load_time:.2f}s")
        
        return llm
        
    except Exception as e:
        print(f"  ✗ Failed to load model: {str(e)}")
        sys.exit(1)


def benchmark_batch_size(llm: LLM, prompts: List[str], batch_size: int) -> Dict[str, float]:
    """Run benchmark for a specific batch size."""
    # Split prompts into batches
    batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    
    latencies = []
    total_samples = 0
    
    start_total = time.perf_counter()
    
    for batch in batches:
        start_batch = time.perf_counter()
        
        # Run classification
        outputs = llm.classify(batch)
        
        batch_latency = (time.perf_counter() - start_batch) * 1000  # Convert to ms
        latencies.append(batch_latency)
        total_samples += len(batch)
    
    total_time = time.perf_counter() - start_total
    
    # Calculate metrics
    throughput = total_samples / total_time
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    return {
        "batch_size": batch_size,
        "total_samples": total_samples,
        "total_time": round(total_time, 3),
        "throughput": round(throughput, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "p50_latency_ms": round(p50_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "p99_latency_ms": round(p99_latency, 2),
    }


def run_benchmarks(model_id: str, 
                   quantization: str, 
                   batch_sizes: List[int],
                   num_samples: int = 1000,
                   balance_lengths: bool = True,
                   output_dir: str = "./results/local_benchmarks") -> List[Dict]:
    """Run comprehensive benchmarks."""
    print("\n" + "=" * 70)
    print(f"Starting Benchmarks")
    print("=" * 70)
    print(f"Model: {model_id}")
    print(f"Quantization: {quantization}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Number of samples: {num_samples}")
    print(f"Balance sequence lengths: {balance_lengths}")
    print("=" * 70)
    
    # Load test data
    prompts, true_labels, id2label = load_test_data(
        num_samples=num_samples,
        balance_lengths=balance_lengths
    )
    
    # Initialize model
    llm = initialize_model(model_id, quantization)
    
    # Run benchmarks for each batch size
    results = []
    
    print(f"\n{'=' * 70}")
    print(f"Running Benchmarks")
    print(f"{'=' * 70}")
    
    for batch_size in batch_sizes:
        print(f"\n[Batch Size: {batch_size}]")
        
        metrics = benchmark_batch_size(llm, prompts, batch_size)
        metrics['quantization'] = quantization
        metrics['model_id'] = model_id
        
        results.append(metrics)
        
        # Print results
        print(f"  Throughput:    {metrics['throughput']:>7.2f} samples/s")
        print(f"  Avg Latency:   {metrics['avg_latency_ms']:>7.2f} ms")
        print(f"  P50 Latency:   {metrics['p50_latency_ms']:>7.2f} ms")
        print(f"  P95 Latency:   {metrics['p95_latency_ms']:>7.2f} ms")
        print(f"  P99 Latency:   {metrics['p99_latency_ms']:>7.2f} ms")
    
    # Save results
    output_path = Path(output_dir) / quantization
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"✓ Results saved to: {results_file}")
    print(f"{'=' * 70}")
    
    # Print summary table
    print("\nSummary Table:")
    print(f"{'Batch Size':<12} {'Throughput':<15} {'Avg Latency':<15} {'P95 Latency':<15}")
    print("-" * 70)
    
    for r in results:
        print(
            f"{r['batch_size']:<12} "
            f"{r['throughput']:<15.2f} "
            f"{r['avg_latency_ms']:<15.2f} "
            f"{r['p95_latency_ms']:<15.2f}"
        )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM classification with different configurations"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="codefactory4791/intent-classification-qwen",
        help="Hugging Face model ID"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "bitsandbytes", "awq", "gptq"],
        help="Quantization method"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8,16,32",
        help="Comma-separated list of batch sizes to test"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to use for benchmarking"
    )
    parser.add_argument(
        "--balance-lengths",
        action="store_true",
        default=True,
        help="Balance dataset with short, medium, and long sequences"
    )
    parser.add_argument(
        "--no-balance-lengths",
        action="store_false",
        dest="balance_lengths",
        help="Disable length balancing (use random sampling)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/local_benchmarks",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    
    # Run benchmarks
    results = run_benchmarks(
        model_id=args.model_id,
        quantization=args.quantization,
        batch_sizes=batch_sizes,
        num_samples=args.num_samples,
        balance_lengths=args.balance_lengths,
        output_dir=args.output_dir
    )
    
    print("\nBenchmarking complete!")
    print(f"\nNext steps:")
    print(f"  1. Review results in: {args.output_dir}/{args.quantization}/")
    print(f"  2. Test other quantization methods")
    print(f"  3. Compare results: python scripts/analyze_results.py")


if __name__ == "__main__":
    main()

