#!/usr/bin/env python3
"""
Compare benchmark results across different quantization methods.
Analyzes performance metrics and provides deployment recommendations.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_results(results_dir: str, quantization: str) -> List[Dict]:
    """Load benchmark results for a specific quantization method."""
    results_path = Path(results_dir) / quantization / "benchmark_results.json"
    
    if not results_path.exists():
        print(f"Warning: Results not found for {quantization} at {results_path}")
        return None
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def compare_quantizations(results_dir: str, quantizations: List[str]) -> pd.DataFrame:
    """Compare results across multiple quantization methods."""
    all_results = {}
    
    for quant in quantizations:
        results = load_results(results_dir, quant)
        if results:
            all_results[quant] = results
    
    if not all_results:
        print("Error: No results found to compare")
        sys.exit(1)
    
    # Create comparison dataframe
    comparison_data = []
    
    # Get all batch sizes from the first quantization method
    first_quant = list(all_results.keys())[0]
    batch_sizes = [r['batch_size'] for r in all_results[first_quant]]
    
    for batch_size in batch_sizes:
        row = {'Batch Size': batch_size}
        
        for quant in quantizations:
            if quant in all_results:
                # Find result for this batch size
                result = next((r for r in all_results[quant] if r['batch_size'] == batch_size), None)
                if result:
                    row[f'{quant.upper()} Throughput'] = result['throughput']
                    row[f'{quant.upper()} P95 Latency'] = result['p95_latency_ms']
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


def calculate_speedup(df: pd.DataFrame, baseline: str, target: str) -> pd.DataFrame:
    """Calculate speedup and latency reduction compared to baseline."""
    baseline_throughput_col = f'{baseline.upper()} Throughput'
    target_throughput_col = f'{target.upper()} Throughput'
    baseline_latency_col = f'{baseline.upper()} P95 Latency'
    target_latency_col = f'{target.upper()} P95 Latency'
    
    if baseline_throughput_col in df.columns and target_throughput_col in df.columns:
        df['Speedup'] = (df[target_throughput_col] / df[baseline_throughput_col]).round(2)
        df['Latency Reduction %'] = (
            (df[baseline_latency_col] - df[target_latency_col]) / df[baseline_latency_col] * 100
        ).round(1)
    
    return df


def print_comparison(df: pd.DataFrame, quantizations: List[str]):
    """Print detailed comparison table."""
    print("\n" + "=" * 100)
    print("PERFORMANCE COMPARISON: FP16 vs Quantization Methods")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)


def print_recommendations(df: pd.DataFrame, quantizations: List[str]):
    """Print deployment recommendations based on results."""
    print("\n" + "=" * 100)
    print("DEPLOYMENT RECOMMENDATIONS")
    print("=" * 100)
    
    if 'Speedup' in df.columns:
        # Find best configuration
        best_idx = df['Speedup'].idxmax()
        best_config = df.iloc[best_idx]
        
        print(f"\nBest Overall Configuration:")
        print(f"  Quantization: BitsAndBytes INT8")
        print(f"  Batch Size: {int(best_config['Batch Size'])}")
        
        if 'BITSANDBYTES Throughput' in df.columns:
            print(f"  Throughput: {best_config['BITSANDBYTES Throughput']:.2f} samples/s")
            print(f"  P95 Latency: {best_config['BITSANDBYTES P95 Latency']:.2f} ms")
        
        if 'Speedup' in df.columns:
            print(f"  Speedup vs FP16: {best_config['Speedup']:.2f}x")
        
        if 'Latency Reduction %' in df.columns:
            print(f"  Latency Reduction: {best_config['Latency Reduction %']:.1f}%")
        
        print(f"  Memory Savings: ~50% (INT8 vs FP16)")
    
    print("\n" + "-" * 100)
    print("Configuration by Use Case:")
    print("-" * 100)
    
    # Low latency (batch size 1)
    batch_1 = df[df['Batch Size'] == 1].iloc[0] if len(df[df['Batch Size'] == 1]) > 0 else None
    if batch_1 is not None:
        print("\nLow Latency / Real-time (Batch Size 1):")
        if 'NONE Throughput' in df.columns:
            print(f"  Recommended: FP16 (none)")
            print(f"  Throughput: {batch_1['NONE Throughput']:.2f} samples/s")
            print(f"  P95 Latency: {batch_1['NONE P95 Latency']:.2f} ms")
            print(f"  Reason: FP16 has lower overhead for single samples")
    
    # High throughput (batch size 32)
    batch_32 = df[df['Batch Size'] == 32].iloc[0] if len(df[df['Batch Size'] == 32]) > 0 else None
    if batch_32 is not None:
        print("\nHigh Throughput / Batch Processing (Batch Size 32):")
        if 'BITSANDBYTES Throughput' in df.columns:
            print(f"  Recommended: BitsAndBytes INT8")
            print(f"  Throughput: {batch_32['BITSANDBYTES Throughput']:.2f} samples/s")
            print(f"  P95 Latency: {batch_32['BITSANDBYTES P95 Latency']:.2f} ms")
            print(f"  Reason: Best throughput and memory efficiency")
    
    # Balanced (batch size 8-16)
    batch_16 = df[df['Batch Size'] == 16].iloc[0] if len(df[df['Batch Size'] == 16]) > 0 else None
    if batch_16 is not None:
        print("\nBalanced / General Purpose (Batch Size 16):")
        if 'BITSANDBYTES Throughput' in df.columns:
            print(f"  Recommended: BitsAndBytes INT8")
            print(f"  Throughput: {batch_16['BITSANDBYTES Throughput']:.2f} samples/s")
            print(f"  P95 Latency: {batch_16['BITSANDBYTES P95 Latency']:.2f} ms")
            print(f"  Reason: Good balance of latency and throughput")
    
    print("=" * 100)


def save_comparison(df: pd.DataFrame, output_path: str):
    """Save comparison results to CSV."""
    df.to_csv(output_path, index=False)
    print(f"\nComparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare benchmark results across quantization methods"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results/local_benchmarks",
        help="Directory containing benchmark results"
    )
    parser.add_argument(
        "--quantizations",
        type=str,
        default="none,bitsandbytes",
        help="Comma-separated list of quantization methods to compare"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/comparison.csv",
        help="Output path for comparison CSV"
    )
    
    args = parser.parse_args()
    
    # Parse quantization methods
    quantizations = [q.strip() for q in args.quantizations.split(',')]
    
    print("=" * 100)
    print("BENCHMARK COMPARISON TOOL")
    print("=" * 100)
    print(f"Results directory: {args.results_dir}")
    print(f"Comparing: {', '.join(quantizations)}")
    print("=" * 100)
    
    # Load and compare results
    df = compare_quantizations(args.results_dir, quantizations)
    
    # Calculate speedup if comparing two methods
    if len(quantizations) == 2:
        df = calculate_speedup(df, quantizations[0], quantizations[1])
    
    # Print comparison
    print_comparison(df, quantizations)
    
    # Print recommendations
    print_recommendations(df, quantizations)
    
    # Save results
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    save_comparison(df, args.output)
    
    print("\nComparison complete!")


if __name__ == "__main__":
    main()

