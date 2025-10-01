#!/usr/bin/env python3

import os
import csv
import sys
import glob
from typing import Dict, Optional

def extract_vlmblind_score(csv_path: str) -> Optional[float]:
    """Extract overall score from VLMBlind CSV."""
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            values = next(reader)

            # Find 'overall' column
            if 'overall' in headers:
                overall_idx = headers.index('overall')
                return float(values[overall_idx])
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    return None

def extract_mathvista_score(csv_path: str) -> Optional[float]:
    """Extract overall score from MathVista CSV."""
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            values = next(reader)

            # Find 'acc' column for overall row
            if 'acc' in headers:
                acc_idx = headers.index('acc')
                return float(values[acc_idx])
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    return None

def extract_mathverse_score(csv_path: str) -> Optional[float]:
    """Extract average of Overall column from MathVerse CSV."""
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)

            # Find 'Overall' column
            if 'Overall' not in headers:
                return None

            overall_idx = headers.index('Overall')
            overall_values = []

            # Read all data rows (skip header)
            for row in reader:
                if len(row) > overall_idx and row[overall_idx]:
                    try:
                        overall_values.append(float(row[overall_idx]))
                    except ValueError:
                        continue

            if overall_values:
                return sum(overall_values) / len(overall_values)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    return None

def extract_charxiv_score(csv_path: str) -> Optional[float]:
    """Extract overall score from CharXiv CSV."""
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            values = next(reader)

            # Find 'Overall' column
            if 'Overall' in headers:
                overall_idx = headers.index('Overall')
                return float(values[overall_idx])
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    return None

def extract_vstar_score(csv_path: str) -> Optional[float]:
    """Extract overall score from VStar CSV."""
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            values = next(reader)

            # Find 'Overall' column
            if 'Overall' in headers:
                overall_idx = headers.index('Overall')
                return float(values[overall_idx])
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    return None

def extract_mmmu_score(csv_path: str) -> Optional[float]:
    """Extract overall score from MMMU CSV."""
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)

            # Skip dev row, get validation row
            next(reader)  # Skip dev
            validation_values = next(reader)

            # Find 'Overall' column
            if 'Overall' in headers:
                overall_idx = headers.index('Overall')
                return float(validation_values[overall_idx])
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    return None

def extract_hr_bench_score(csv_path: str, score_type: str) -> Optional[float]:
    """Extract HR-Bench score (all, single, or cross)."""
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)

            # Read all rows to find Average row
            for row in reader:
                if len(row) >= 2 and row[1] == score_type and row[0] == "Average":
                    return float(row[2])  # accuracy column
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_scores.py <model_path>")
        print("Example: python extract_scores.py outputs/vllm-code_v3.1_32B_7B_n8_step48_sp")
        sys.exit(1)

    model_path = sys.argv[1]

    if not os.path.exists(model_path):
        print(f"Error: Path {model_path} does not exist")
        sys.exit(1)

    # Define the dataset mappings and their extraction functions
    datasets = {
        'VLMBlind': ('*VLMBlind_acc.csv', extract_vlmblind_score),
        'MathVista': ('*MathVista_MINI_gpt-4o-mini_score.csv', extract_mathvista_score),
        'CharXiv-reason': ('*CharXiv_reasoning_val_gpt-4o-mini-2024-07-18_acc.csv', extract_charxiv_score),
        'CharXiv-describe': ('*CharXiv_descriptive_val_gpt-4o-mini-2024-07-18_acc.csv', extract_charxiv_score),
        'V*': ('*VStarBench_acc.csv', extract_vstar_score),
        'MMMU': ('*MMMU_DEV_VAL_acc.csv', extract_mmmu_score),
        'MathVerse-Mini': ('*MathVerse_MINI_gpt-4o-mini_score.csv', extract_mathverse_score),
        'MathVision-mini': ('*MathVision_MINI_gpt-4o-mini_score.csv', extract_mathvista_score),
        'HR-4K-all': ('*HRBench4K_acc.csv', lambda path: extract_hr_bench_score(path, 'all')),
        'HR-4K-single': ('*HRBench4K_acc.csv', lambda path: extract_hr_bench_score(path, 'single')),
        'HR-4K-cross': ('*HRBench4K_acc.csv', lambda path: extract_hr_bench_score(path, 'cross')),
        'HR-8K-all': ('*HRBench8K_acc.csv', lambda path: extract_hr_bench_score(path, 'all')),
        'HR-8K-single': ('*HRBench8K_acc.csv', lambda path: extract_hr_bench_score(path, 'single')),
        'HR-8K-cross': ('*HRBench8K_acc.csv', lambda path: extract_hr_bench_score(path, 'cross')),
    }

    # Order as requested by user
    dataset_order = [
        'VLMBlind', 'MathVista', 'CharXiv-reason', 'CharXiv-describe', 'V*', 'MMMU',
        'MathVerse-Mini', 'MathVision-mini', 'HR-4K-all', 'HR-4K-single', 'HR-4K-cross',
        'HR-8K-all', 'HR-8K-single', 'HR-8K-cross'
    ]

    scores = []

    for dataset_name in dataset_order:
        if dataset_name not in datasets:
            scores.append('N/A')
            continue

        pattern, extract_func = datasets[dataset_name]
        csv_files = glob.glob(os.path.join(model_path, pattern))

        if not csv_files:
            scores.append('N/A')
            continue

        # Use the first matching file
        csv_file = csv_files[0]
        score = extract_func(csv_file)

        if score is not None:
            scores.append(f"{score:.4f}")
        else:
            scores.append('N/A')

    # Print scores in requested format
    print(','.join(scores))

if __name__ == "__main__":
    main()