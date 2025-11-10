#!/usr/bin/env python3
"""
Tool Usage & Answer Accuracy Analysis

Pure rule-based analysis (no image inspection).
Analyzes answer accuracy stratified by:
1. Tool use vs no tool use
2. Number of tool calls (0, 1, 2, 3+, etc.)

Usage:
    python analyze_tool_usage.py <rollout_file> <result_xlsx> [output_dir]

Example:
    python analyze_tool_usage.py \\
        outputs/response/vllm-pixelreasoner/VStarBench_20251110041538.jsonl \\
        outputs/vllm-pixelreasoner/vllm-pixelreasoner_VStarBench_gpt-4o-mini_result.xlsx
"""

import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd


def extract_model_name(filepath: str) -> str:
    """
    Extract model name from file path.

    Example: outputs/response/vllm-pixelreasoner/... -> pixelreasoner
    """
    path = Path(filepath)
    dir_name = path.parent.name
    model_name = dir_name.replace('vllm-', '').replace('outputs/', '')
    return model_name


def load_jsonl(filepath: str) -> List[dict]:
    """Load JSONL file into list of dicts."""
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]


def extract_index_from_image_path(image_path: str) -> Optional[int]:
    """
    Extract index from image path.

    Example: /path/to/VStarBench/9.jpg -> 9
    """
    if not image_path:
        return None

    filename = Path(image_path).stem
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))

    return None


def load_answer_accuracy(xlsx_file: str) -> Dict[int, Dict[str, Any]]:
    """
    Load answer accuracy from Excel result file.

    Returns:
        Dict mapping index -> {prediction, answer, correct, ...}
    """
    try:
        df = pd.read_excel(xlsx_file)
    except ImportError:
        print("Warning: openpyxl not installed. Trying to read as CSV...")
        csv_file = xlsx_file.replace('.xlsx', '.csv')
        if Path(csv_file).exists():
            df = pd.read_csv(csv_file)
        else:
            print("Error: Could not load result file.")
            return {}

    result_map = {}
    for _, row in df.iterrows():
        idx = row.get('index')
        if pd.notna(idx):
            result_map[int(idx)] = {
                'prediction': row.get('prediction', None),
                'answer': row.get('answer', None),
                'correct': bool(row.get('hit', False)) if pd.notna(row.get('hit')) else None
            }

    return result_map


def extract_question_and_image(entry: dict) -> tuple[str, str]:
    """Extract question text and image path from entry."""
    for msg in entry:
        if msg['role'] == 'user':
            content = msg.get('content', [])

            image_path = None
            question_text = None

            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'image_url':
                        image_path = item.get('image_url', '')
                    elif item.get('type') == 'text':
                        text = item.get('text', '')
                        match = re.search(r'Question: (.+?)(?:\n|Options:)', text, re.DOTALL)
                        if match:
                            question_text = match.group(1).strip()

            if question_text and image_path:
                return question_text, image_path

    return "", ""


def count_tool_calls(entry: dict) -> int:
    """
    Count number of tool calls in entry.

    Detects both:
    - pixelreasoner: <tool_call> tags
    - thyme/codev3.4: <code> blocks with Python
    """
    count = 0
    for msg in entry:
        if msg['role'] == 'assistant':
            content = msg.get('content', '')
            if isinstance(content, str):
                # Count <tool_call> tags (pixelreasoner)
                count += content.count('<tool_call>')

                # Count <code> blocks (thyme/codev3.4)
                count += content.count('<code>')
    return count


# ============================================================================
# Tool Usage Analysis
# ============================================================================

def analyze_tool_usage(entries_with_answer: List[dict]) -> Dict[str, Any]:
    """
    Analyze answer accuracy stratified by tool usage.

    Returns:
        Dict with tool usage statistics
    """
    # Stratify by number of tool calls
    by_tool_count = {}

    for entry in entries_with_answer:
        num_tools = entry.get('num_tools', 0)

        if num_tools not in by_tool_count:
            by_tool_count[num_tools] = {
                'total': 0,
                'correct': 0
            }

        by_tool_count[num_tools]['total'] += 1
        if entry.get('answer_correct', False):
            by_tool_count[num_tools]['correct'] += 1

    # Calculate stats
    tool_stats = {}
    for num_tools, counts in sorted(by_tool_count.items()):
        accuracy = counts['correct'] / counts['total'] * 100 if counts['total'] > 0 else 0
        tool_stats[num_tools] = {
            'total': counts['total'],
            'correct': counts['correct'],
            'accuracy': accuracy
        }

    # Overall stats
    total_entries = len(entries_with_answer)
    with_tools = sum(counts['total'] for n, counts in by_tool_count.items() if n > 0)
    without_tools = by_tool_count.get(0, {}).get('total', 0)

    with_tools_correct = sum(
        counts['correct'] for n, counts in by_tool_count.items() if n > 0
    )
    without_tools_correct = by_tool_count.get(0, {}).get('correct', 0)

    return {
        'total_entries': total_entries,
        'with_tools': with_tools,
        'without_tools': without_tools,
        'with_tools_accuracy': with_tools_correct / with_tools * 100 if with_tools > 0 else 0,
        'without_tools_accuracy': without_tools_correct / without_tools * 100 if without_tools > 0 else 0,
        'by_tool_count': tool_stats
    }


def print_tool_usage_stats(stats: Dict[str, Any]):
    """Print tool usage statistics in a readable format."""
    print("\n" + "="*80)
    print("TOOL USAGE & ANSWER ACCURACY ANALYSIS")
    print("="*80)

    total = stats['total_entries']
    with_tools = stats['with_tools']
    without_tools = stats['without_tools']

    print(f"\nTotal Entries: {total}")
    print()

    # Tool use vs no tool use
    print("TOOL USAGE OVERVIEW:")
    print("-"*80)
    print(f"With Tools:     {with_tools:>4} ({with_tools/total*100:>5.1f}%)   "
          f"Accuracy: {stats['with_tools_accuracy']:>5.1f}%")
    print(f"Without Tools:  {without_tools:>4} ({without_tools/total*100:>5.1f}%)   "
          f"Accuracy: {stats['without_tools_accuracy']:>5.1f}%")

    # Tool usage distribution (among those who used tools)
    if with_tools > 0:
        print("\n" + "-"*80)
        print("TOOL USAGE DISTRIBUTION (among tool users):")
        print("-"*80)
        for num_tools, tool_stats in sorted(stats['by_tool_count'].items()):
            if num_tools > 0:  # Only show tool users
                pct_of_tool_users = tool_stats['total'] / with_tools * 100
                print(f"  Used {num_tools} time(s):  {tool_stats['total']:>4} ({pct_of_tool_users:>5.1f}% of tool users)")

    # Stratified by number of tool calls with accuracy
    print("\n" + "-"*80)
    print("ACCURACY BY NUMBER OF TOOL CALLS:")
    print("-"*80)
    print(f"{'# Tools':<12} {'Count':<8} {'% of Total':<12} {'Correct':<10} {'Accuracy':<12}")
    print("-"*80)

    for num_tools, tool_stats in sorted(stats['by_tool_count'].items()):
        label = f"{num_tools} tool(s)" if num_tools > 0 else "No tools"
        pct_of_total = tool_stats['total'] / total * 100
        print(f"{label:<12} {tool_stats['total']:<8} {pct_of_total:>5.1f}%       "
              f"{tool_stats['correct']:<10} {tool_stats['accuracy']:>5.1f}%")


# ============================================================================
# Main Analysis
# ============================================================================

def main(rollout_file: str, result_xlsx: str, output_dir: str = None):
    """
    Main analysis function.

    Args:
        rollout_file: Path to rollout JSONL file
        result_xlsx: Path to result Excel file with answer accuracy
        output_dir: Output directory (default: auto-generated from model name)
    """
    # Extract model name and create output directory
    if output_dir is None:
        model_name = extract_model_name(rollout_file)
        output_dir = f"tool_usage_{model_name}"

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Define output files
    results_file = output_path / "tool_usage_results.jsonl"
    stats_file = output_path / "tool_usage_stats.json"

    print("="*80)
    print("TOOL USAGE & ANSWER ACCURACY ANALYSIS")
    print("="*80)
    print(f"Rollout File: {rollout_file}")
    print(f"Result XLSX:  {result_xlsx}")
    print(f"Output Dir:   {output_dir}")
    print("="*80)

    # Load answer accuracy data
    print("\nLoading answer accuracy from Excel...")
    answer_map = load_answer_accuracy(result_xlsx)
    print(f"  Loaded {len(answer_map)} answer records")

    # Load rollout data
    print("\nLoading rollout data...")
    data = load_jsonl(rollout_file)
    print(f"  Loaded {len(data)} entries")

    # Extract tool usage and match with answers
    print("\nAnalyzing tool usage...")
    entries_with_info = []

    for idx, entry in enumerate(data):
        question, image_path = extract_question_and_image(entry)
        num_tools = count_tool_calls(entry)

        # Extract index from image path for matching
        image_idx = extract_index_from_image_path(image_path)

        # Get answer accuracy if available
        answer_info = answer_map.get(image_idx, {}) if image_idx is not None else {}

        entry_info = {
            'entry_idx': idx,
            'image_index': image_idx,
            'question': question,
            'image_path': image_path,
            'num_tools': num_tools,
            'prediction': answer_info.get('prediction'),
            'answer': answer_info.get('answer'),
            'answer_correct': answer_info.get('correct')
        }

        entries_with_info.append(entry_info)

    # Filter to entries with answer info
    entries_with_answer = [e for e in entries_with_info if e['answer_correct'] is not None]
    print(f"  Found {len(entries_with_answer)} entries with answer data")

    # Analyze tool usage
    print("\nAnalyzing tool usage patterns...")
    tool_stats = analyze_tool_usage(entries_with_answer)

    # Save detailed results
    print("\nSaving results...")
    with open(results_file, 'w') as f:
        for entry in entries_with_info:
            f.write(json.dumps(entry) + '\n')
    print(f"  Saved to: {results_file}")

    # Save stats
    with open(stats_file, 'w') as f:
        json.dump(tool_stats, f, indent=2)
    print(f"  Saved to: {stats_file}")

    # Print statistics
    print_tool_usage_stats(tool_stats)

    print("\n" + "="*80)
    print(f"All results saved to: {output_dir}/")
    print(f"  - tool_usage_results.jsonl")
    print(f"  - tool_usage_stats.json")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_tool_usage.py <rollout_file> <result_xlsx> [output_dir]")
        print("\nExample:")
        print("  python analyze_tool_usage.py \\")
        print("    outputs/response/vllm-pixelreasoner/VStarBench_20251110041538.jsonl \\")
        print("    outputs/vllm-pixelreasoner/vllm-pixelreasoner_VStarBench_gpt-4o-mini_result.xlsx")
        sys.exit(1)

    rollout_file = sys.argv[1]
    result_xlsx = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None

    main(rollout_file, result_xlsx, output_dir)
