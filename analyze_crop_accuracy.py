#!/usr/bin/env python3
"""
Crop Accuracy Analysis

Analyzes whether cropped images contain the objects mentioned in the question.
Uses GPT-4o Vision to judge each crop. Also loads answer accuracy from result Excel.

Features:
- Judges each crop for whether it contains relevant visual content
- Analyzes correlation between crop accuracy and answer correctness
- Automatically copies 30 sample images (originals + crops) for manual inspection

Output Structure:
- crop_accuracy_{model}/{dataset}/
  - crop_accuracy_results.jsonl
  - crop_accuracy_confusion_matrix.json
  - {image_index}/          # Sample directories (30 samples)
    - original.jpg           # Original image
    - crops/                 # Folder with all crops
      - crop_0.jpg
      - crop_1.jpg
      - ...
    - metadata.json          # Question, judgments, accuracy

Usage:
    python analyze_crop_accuracy.py <rollout_file> <result_xlsx> [output_dir]

Example:
    python analyze_crop_accuracy.py \\
        outputs/response/vllm-pixelreasoner/VStarBench_20251110041538.jsonl \\
        outputs/vllm-pixelreasoner/vllm-pixelreasoner_VStarBench_gpt-4o-mini_result.xlsx
"""

import asyncio
import json
import logging
import sys
import re
import base64
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI
from tqdm import tqdm
import pandas as pd

# logging.basicConfig(level=logging.INFO)


def extract_model_name(filepath: str) -> str:
    """
    Extract model name from file path.

    Example: outputs/response/vllm-pixelreasoner/... -> pixelreasoner
    """
    path = Path(filepath)

    # Get directory name (e.g., vllm-pixelreasoner)
    dir_name = path.parent.name

    # Remove common prefixes
    model_name = dir_name.replace('vllm-', '').replace('outputs/', '')

    return model_name


def extract_dataset_name(filepath: str) -> str:
    """
    Extract dataset name from rollout file path.

    Example: VStarBench_20251110062641.jsonl -> VStarBench
    Example: HRBench8K_20251110160827.jsonl -> HRBench8K
    """
    path = Path(filepath)
    filename = path.stem  # Get filename without extension
    # Split by underscore and take everything before the timestamp
    # Timestamp format: YYYYMMDDHHMMSS (14 digits)
    parts = filename.split('_')
    # Find where timestamp starts (look for part with all digits and length >= 8)
    dataset_parts = []
    for part in parts:
        if part.isdigit() and len(part) >= 8:  # Timestamp detected
            break
        dataset_parts.append(part)

    return '_'.join(dataset_parts) if dataset_parts else filename

# Configuration
OPENAI_API_KEY = ""
GPT_MODEL = "gpt-4o"

# Create async client for OpenAI
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


# ============================================================================
# Data Extraction
# ============================================================================

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.

    Args:
        image_path: Path to image file

    Returns:
        Base64 encoded string with data URI prefix
    """
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')

    # Determine image format from extension
    ext = Path(image_path).suffix.lower()
    if ext in ['.jpg', '.jpeg']:
        mime_type = "image/jpeg"
    elif ext == '.png':
        mime_type = "image/png"
    elif ext == '.gif':
        mime_type = "image/gif"
    elif ext == '.webp':
        mime_type = "image/webp"
    else:
        mime_type = "image/jpeg"  # default

    return f"data:{mime_type};base64,{encoded}"


def load_jsonl(filepath: str) -> List[dict]:
    """Load JSONL file into list of dicts."""
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]


def extract_index_from_image_path(image_path: str) -> Optional[int]:
    """
    Extract index from image path.

    Example: /path/to/VStarBench/9.jpg -> 9

    Args:
        image_path: Path to image file

    Returns:
        Index as integer, or None if not found
    """
    if not image_path:
        return None

    # Get filename without extension
    filename = Path(image_path).stem

    # Try to extract number from filename
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))

    return None


def load_answer_accuracy(xlsx_file: str) -> Dict[int, Dict[str, Any]]:
    """
    Load answer accuracy from Excel result file.

    Args:
        xlsx_file: Path to Excel result file

    Returns:
        Dict mapping index -> {prediction, answer, correct, ...}
    """
    try:
        df = pd.read_excel(xlsx_file)
    except ImportError:
        logging.warning("openpyxl not installed. Trying to read as CSV...")
        # Try CSV version
        csv_file = xlsx_file.replace('.xlsx', '.csv')
        if Path(csv_file).exists():
            df = pd.read_csv(csv_file)
        else:
            logging.error("Could not load result file. Please install openpyxl or provide CSV version.")
            return {}

    # Create mapping from index to result
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
    """
    Extract question text and original image path from entry.

    Returns:
        (question_text, image_path)
    """
    # Find user message with question
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
                        # Extract question
                        match = re.search(r'Question: (.+?)(?:\n|Options:)', text, re.DOTALL)
                        if match:
                            question_text = match.group(1).strip()

            if question_text and image_path:
                return question_text, image_path

    return "", ""


def has_tool_usage(entry: dict) -> bool:
    """
    Check if entry has tool usage based on message structure.

    Tool usage is indicated by:
    1. More than 3 messages (system, user, assistant with tool, user with tool return, assistant)
    2. More than 1 user message (original + tool return)

    Returns:
        True if tools were used, False otherwise
    """
    if len(entry) > 3:
        return True

    user_count = sum(1 for msg in entry if msg['role'] == 'user')
    return user_count > 1


def extract_crops(entry: dict) -> List[str]:
    """
    Extract all cropped image paths from tool responses.

    Logic:
    - entry[0] = system
    - entry[1] = user with original image
    - entry[2+] = any user messages with images are crops

    Returns:
        List of crop image paths
    """
    crops = []

    # Skip first two messages (system and original user message)
    # Any images in subsequent user messages are crops
    for idx, msg in enumerate(entry):
        if idx <= 1:  # Skip system (0) and first user (1)
            continue

        if msg['role'] == 'user':
            content = msg.get('content', [])
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'image_url':
                    url = item.get('image_url', '')
                    crops.append(url)

    return crops


# ============================================================================
# LLM Judge for Crop Accuracy
# ============================================================================

JUDGE_SYSTEM_PROMPT = """You are evaluating whether an image contains the relevant visual content to answer a question.

Your task: Determine if the image clearly shows the objects/content mentioned or implied in the question, and if those objects are the main focus of the image.

Respond with JSON:
{
  "contains_object": 1 or 0,
  "confidence": "high/medium/low",
  "reasoning": "brief explanation"
}

Use 1 if:
- The objects/content needed to answer the question are clearly visible
- They are the main focus/subject of the image
- The image provides clear, focused visual information relevant to answering the question

Use 0 if:
- The relevant objects are not visible, unclear, only partially captured, or very small
- The objects are not the main subject of the image
- The image does not provide clear visual information to answer the question"""


def create_judge_prompt(question: str) -> str:
    """Create prompt for crop accuracy judge."""
    return f"""Question: {question}

Does this image clearly show the objects/content needed to answer this question, and are they the main focus of the image?

Evaluate if the image provides clear, focused visual information relevant to answering the question.
Answer with 1 (yes, relevant content is clearly visible and main focus) or 0 (no, relevant content is not visible, unclear, partial, or not the main subject)."""


# ============================================================================
# Async Processing
# ============================================================================

@dataclass
class CropJudgeRequest:
    """Single crop judgment request"""
    question: str
    crop_image: str
    crop_index: int
    request_id: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CropJudgeResponse:
    """Single crop judgment response"""
    request_id: str
    crop_index: int
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CropAccuracyJudge:
    """Async processor for crop accuracy judgments"""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = GPT_MODEL,
        max_concurrent: int = 20,
        temperature: float = 0.0,
        max_tokens: int = 500,
        retries: int = 3
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single(self, request: CropJudgeRequest) -> CropJudgeResponse:
        """Process a single crop judgment with retries"""
        prompt = create_judge_prompt(request.question)

        # Encode image to base64
        try:
            if request.crop_image.startswith("http"):
                assert False
                # Use URL directly
                image_data = request.crop_image
            else:
                # Encode local file to base64
                image_data = encode_image_to_base64(request.crop_image)
        except Exception as e:
            logging.error(f"[{request.request_id}] Failed to encode image: {e}")
            return CropJudgeResponse(
                request_id=request.request_id,
                crop_index=request.crop_index,
                error=f"Image encoding failed: {e}",
                metadata=request.metadata
            )

        for attempt in range(1, self.retries + 1):
            try:
                async with self.semaphore:
                    # Prepare messages with image
                    messages = [
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_data
                                    }
                                }
                            ]
                        }
                    ]

                    # Add 2 minute timeout
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            response_format={"type": "json_object"},
                            temperature=self.temperature,
                            max_tokens=self.max_tokens
                        ),
                        timeout=120.0  # 2 minute timeout
                    )

                # Parse JSON response
                content = response.choices[0].message.content
                parsed = json.loads(content)

                return CropJudgeResponse(
                    request_id=request.request_id,
                    crop_index=request.crop_index,
                    response=parsed,
                    metadata=request.metadata
                )

            except asyncio.TimeoutError:
                logging.error(
                    f"[{request.request_id}] Crop {request.crop_index} Attempt {attempt}/{self.retries} timed out after 2 minutes"
                )
                if attempt < self.retries:
                    await asyncio.sleep(2 ** (attempt - 1))
                else:
                    return CropJudgeResponse(
                        request_id=request.request_id,
                        crop_index=request.crop_index,
                        error="Timeout after 2 minutes",
                        metadata=request.metadata
                    )
            except Exception as e:
                logging.error(
                    f"[{request.request_id}] Crop {request.crop_index} Attempt {attempt}/{self.retries} failed: {e}"
                )
                if attempt < self.retries:
                    await asyncio.sleep(2 ** (attempt - 1))
                else:
                    return CropJudgeResponse(
                        request_id=request.request_id,
                        crop_index=request.crop_index,
                        error=str(e),
                        metadata=request.metadata
                    )

    async def process_batch(
        self,
        requests: List[CropJudgeRequest],
        show_progress: bool = True
    ) -> List[CropJudgeResponse]:
        """Process multiple requests in parallel"""
        tasks = [self.process_single(req) for req in requests]

        if show_progress:
            results = []
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Judging Crops"):
                results.append(await coro)
            return results
        else:
            return await asyncio.gather(*tasks)


# ============================================================================
# Confusion Matrix Analysis
# ============================================================================

def analyze_confusion_matrix(results: List[dict]) -> Dict[str, Any]:
    """
    Create confusion matrix of crop accuracy vs answer correctness.

    Returns:
        Dict with confusion matrix and statistics
    """
    # Filter to entries with both crop judgments and answer info
    valid_entries = [
        r for r in results
        if r.get('answer_correct') is not None and r.get('num_crops', 0) > 0
    ]

    # Initialize confusion matrix
    matrix = {
        'crop_correct_answer_correct': 0,
        'crop_correct_answer_wrong': 0,
        'crop_wrong_answer_correct': 0,
        'crop_wrong_answer_wrong': 0
    }

    # Populate matrix
    for entry in valid_entries:
        any_crop_correct = entry.get('any_crop_correct', False)
        answer_correct = entry.get('answer_correct', False)

        if any_crop_correct and answer_correct:
            matrix['crop_correct_answer_correct'] += 1
        elif any_crop_correct and not answer_correct:
            matrix['crop_correct_answer_wrong'] += 1
        elif not any_crop_correct and answer_correct:
            matrix['crop_wrong_answer_correct'] += 1
        else:  # not any_crop_correct and not answer_correct
            matrix['crop_wrong_answer_wrong'] += 1

    total = len(valid_entries)

    # Calculate statistics
    stats = {
        'total_entries': len(results),
        'valid_entries': total,
        'matrix': matrix,
        'matrix_pct': {
            key: (value / total * 100) if total > 0 else 0
            for key, value in matrix.items()
        }
    }

    # Additional insights
    if total > 0:
        crop_correct_total = matrix['crop_correct_answer_correct'] + matrix['crop_correct_answer_wrong']
        crop_wrong_total = matrix['crop_wrong_answer_correct'] + matrix['crop_wrong_answer_wrong']
        answer_correct_total = matrix['crop_correct_answer_correct'] + matrix['crop_wrong_answer_correct']
        answer_wrong_total = matrix['crop_correct_answer_wrong'] + matrix['crop_wrong_answer_wrong']

        stats['insights'] = {
            'crop_accuracy_rate': crop_correct_total / total * 100,
            'answer_accuracy_rate': answer_correct_total / total * 100,
            'answer_correct_given_crop_correct': (
                matrix['crop_correct_answer_correct'] / crop_correct_total * 100
                if crop_correct_total > 0 else 0
            ),
            'answer_correct_given_crop_wrong': (
                matrix['crop_wrong_answer_correct'] / crop_wrong_total * 100
                if crop_wrong_total > 0 else 0
            )
        }

    return stats


def print_confusion_matrix(stats: Dict[str, Any]):
    """Print confusion matrix in a readable format."""
    matrix = stats['matrix']
    matrix_pct = stats['matrix_pct']
    total = stats['valid_entries']

    print("\n" + "="*80)
    print("CROP-ANSWER CONFUSION MATRIX")
    print("="*80)
    print(f"\nTotal Valid Entries: {total}")
    print()

    # Print matrix
    print("                        Answer Correct    Answer Wrong      Total")
    print("-"*80)

    cc_ac = matrix['crop_correct_answer_correct']
    cc_aw = matrix['crop_correct_answer_wrong']
    cc_total = cc_ac + cc_aw

    cw_ac = matrix['crop_wrong_answer_correct']
    cw_aw = matrix['crop_wrong_answer_wrong']
    cw_total = cw_ac + cw_aw

    print(f"Crop Correct (Any)      {cc_ac:>4} ({matrix_pct['crop_correct_answer_correct']:>5.1f}%)   "
          f"{cc_aw:>4} ({matrix_pct['crop_correct_answer_wrong']:>5.1f}%)   "
          f"{cc_total:>4} ({cc_total/total*100:>5.1f}%)")

    print(f"Crop Wrong (All)        {cw_ac:>4} ({matrix_pct['crop_wrong_answer_correct']:>5.1f}%)   "
          f"{cw_aw:>4} ({matrix_pct['crop_wrong_answer_wrong']:>5.1f}%)   "
          f"{cw_total:>4} ({cw_total/total*100:>5.1f}%)")

    ac_total = cc_ac + cw_ac
    aw_total = cc_aw + cw_aw

    print("-"*80)
    print(f"Total                   {ac_total:>4} ({ac_total/total*100:>5.1f}%)   "
          f"{aw_total:>4} ({aw_total/total*100:>5.1f}%)   "
          f"{total:>4} (100.0%)")

    # Print insights
    if 'insights' in stats:
        insights = stats['insights']
        print("\n" + "-"*80)
        print("INSIGHTS")
        print("-"*80)
        print(f"\nOverall Rates:")
        print(f"  Crop Accuracy (Any Correct):  {insights['crop_accuracy_rate']:>5.1f}%")
        print(f"  Answer Accuracy:              {insights['answer_accuracy_rate']:>5.1f}%")

        print(f"\nConditional Answer Accuracy:")
        print(f"  Given Crop Correct:           {insights['answer_correct_given_crop_correct']:>5.1f}%")
        print(f"  Given Crop Wrong:             {insights['answer_correct_given_crop_wrong']:>5.1f}%")

        # Calculate lift
        if insights['answer_correct_given_crop_wrong'] > 0:
            lift = insights['answer_correct_given_crop_correct'] / insights['answer_correct_given_crop_wrong']
            print(f"\nLift (Correct Crop → Correct Answer): {lift:.2f}x")


def copy_sample_images(output_records: List[dict], output_dir: Path, num_samples: int = 30):
    """
    Copy sample original images and their crops to the output directory for inspection.

    Args:
        output_records: List of output records with crop information
        output_dir: Base output directory
        num_samples: Number of samples to copy (default: 30)
    """
    print(f"\nCopying {num_samples} sample images to output directory...")

    # Filter to entries with crops
    entries_with_crops = [r for r in output_records if r['num_crops'] > 0]

    # Select samples (first N entries with crops)
    samples = entries_with_crops[:num_samples]

    copied_count = 0
    for record in tqdm(samples, desc="Copying samples"):
        image_idx = record['image_index']
        if image_idx is None:
            continue

        # Create directory for this sample: {output_dir}/{image_index}/
        sample_dir = output_dir / str(image_idx)
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Create crops subdirectory
        crops_dir = sample_dir / "crops"
        crops_dir.mkdir(exist_ok=True)

        # Copy original image
        original_path = Path(record['original_image'])
        if original_path.exists():
            # Keep original extension
            dest_original = sample_dir / f"original{original_path.suffix}"
            try:
                shutil.copy2(original_path, dest_original)
            except Exception as e:
                logging.warning(f"Failed to copy original image {original_path}: {e}")
                continue

        # Copy crop images
        for crop_idx, crop_path in enumerate(record['crop_paths']):
            crop_path_obj = Path(crop_path)
            if crop_path_obj.exists():
                # Name crops as crop_0.jpg, crop_1.jpg, etc.
                dest_crop = crops_dir / f"crop_{crop_idx}{crop_path_obj.suffix}"
                try:
                    shutil.copy2(crop_path_obj, dest_crop)
                except Exception as e:
                    logging.warning(f"Failed to copy crop {crop_path}: {e}")

        # Save metadata for this sample
        metadata = {
            "image_index": image_idx,
            "question": record["question"],
            "num_crops": record["num_crops"],
            "crop_judgments": record["crop_judgments"],
            "any_crop_correct": record["any_crop_correct"],
            "answer_correct": record.get("answer_correct"),
            "prediction": record.get("prediction"),
            "answer": record.get("answer"),
            "crop_details": record["crop_details"]
        }
        metadata_file = sample_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        copied_count += 1

    print(f"  Copied {copied_count} samples to {output_dir}/")
    print(f"  Each sample directory contains:")
    print(f"    - original.jpg (or .png, etc.)")
    print(f"    - crops/ (folder with crop_0.jpg, crop_1.jpg, etc.)")
    print(f"    - metadata.json (question, judgments, accuracy)")


# ============================================================================
# Main Analysis
# ============================================================================

async def main(rollout_file: str, result_xlsx: str, output_dir: str = None):
    """
    Main analysis function.

    Args:
        rollout_file: Path to rollout JSONL file
        result_xlsx: Path to result Excel file with answer accuracy
        output_dir: Output directory (default: auto-generated from model and dataset name)
    """
    # Extract model name, dataset name, and create output directory
    if output_dir is None:
        model_name = extract_model_name(rollout_file)
        dataset_name = extract_dataset_name(rollout_file)
        output_dir = f"crop_accuracy_{model_name}/{dataset_name}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define output files
    results_file = output_path / "crop_accuracy_results.jsonl"
    confusion_file = output_path / "crop_accuracy_confusion_matrix.json"

    print("="*80)
    print("CROP ACCURACY ANALYSIS")
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

    # Extract crops and create judge requests
    print("\nAnalyzing entries and creating judge requests...")
    judge_requests = []
    entry_metadata = []

    for idx, entry in enumerate(data):
        question, image_path = extract_question_and_image(entry)
        crops = extract_crops(entry)
        used_tools = has_tool_usage(entry)

        # Extract index from image path for matching with answer accuracy
        image_idx = extract_index_from_image_path(image_path)

        # Get answer accuracy if available
        answer_info = answer_map.get(image_idx, {}) if image_idx is not None else {}

        entry_meta = {
            "entry_idx": idx,
            "image_index": image_idx,
            "question": question,
            "original_image": image_path,
            "num_crops": len(crops),
            "crop_paths": crops,
            "has_tool_usage": used_tools,
            "prediction": answer_info.get('prediction'),
            "answer": answer_info.get('answer'),
            "answer_correct": answer_info.get('correct')
        }
        entry_metadata.append(entry_meta)

        # Create judge request for each crop
        for crop_idx, crop_path in enumerate(crops):
            judge_requests.append(CropJudgeRequest(
                question=question,
                crop_image=crop_path,
                crop_index=crop_idx,
                request_id=f"entry_{idx}_crop_{crop_idx}",
                metadata={"entry_idx": idx, "crop_idx": crop_idx}
            ))

    print(f"  Found {len(judge_requests)} crops across {len(data)} entries")
    print(f"  Entries with crops: {sum(1 for m in entry_metadata if m['num_crops'] > 0)}")

    if len(judge_requests) == 0:
        print("\n[WARNING] No crops found to judge!")
        return

    # Process with LLM judge
    print(f"\nProcessing with {GPT_MODEL} vision judge...")
    judge = CropAccuracyJudge(
        client=client,
        model=GPT_MODEL,
        max_concurrent=20,
        temperature=0.0
    )

    results = await judge.process_batch(judge_requests)

    # Aggregate results by entry
    print("\nAggregating results...")
    entry_results = {}

    for result in results:
        if result.response:
            entry_idx = result.metadata.get("entry_idx")
            crop_idx = result.metadata.get("crop_idx")

            if entry_idx not in entry_results:
                entry_results[entry_idx] = []

            entry_results[entry_idx].append({
                "crop_idx": crop_idx,
                "contains_object": result.response.get("contains_object", 0),
                "confidence": result.response.get("confidence", "unknown"),
                "reasoning": result.response.get("reasoning", "")
            })

    # Create output
    print("\nCreating output...")
    output_records = []

    for idx, meta in enumerate(entry_metadata):
        crop_judgments = entry_results.get(idx, [])

        # Sort by crop_idx
        crop_judgments.sort(key=lambda x: x["crop_idx"])

        # Extract judgment list
        judge_list = [j["contains_object"] for j in crop_judgments]

        # Determine if entry is "faithful":
        # - Has tool usage structure but no crops generated (tool attempted but failed)
        is_faithful = meta["has_tool_usage"] and meta["num_crops"] == 0

        record = {
            "entry_idx": idx,
            "image_index": meta["image_index"],
            "question": meta["question"],
            "original_image": meta["original_image"],
            "num_crops": meta["num_crops"],
            "crop_paths": meta["crop_paths"],
            "crop_judgments": judge_list,
            "crop_details": crop_judgments,
            "mean_crop_accuracy": sum(judge_list) / len(judge_list) if judge_list else 0.0,
            "max_crop_accuracy": max(judge_list) if judge_list else 0,
            "all_crops_correct": all(j == 1 for j in judge_list) if judge_list else False,
            "any_crop_correct": any(j == 1 for j in judge_list) if judge_list else False,
            "faithful": is_faithful,
            "has_tool_usage": meta["has_tool_usage"],
            "prediction": meta["prediction"],
            "answer": meta["answer"],
            "answer_correct": meta["answer_correct"]
        }

        output_records.append(record)

    # Save results to JSONL
    print("\nSaving results...")
    with open(results_file, 'w') as f:
        for record in output_records:
            f.write(json.dumps(record) + '\n')
    print(f"  Saved to: {results_file}")

    # Analyze and save confusion matrix
    print("\nAnalyzing confusion matrix...")
    confusion_stats = analyze_confusion_matrix(output_records)

    with open(confusion_file, 'w') as f:
        json.dump(confusion_stats, f, indent=2)
    print(f"  Saved to: {confusion_file}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    total_entries = len(output_records)
    entries_with_crops = sum(1 for r in output_records if r["num_crops"] > 0)
    total_crops = sum(r["num_crops"] for r in output_records)
    faithful_entries = sum(1 for r in output_records if r["faithful"])
    no_tool_entries = sum(1 for r in output_records if not r["has_tool_usage"])

    print(f"Total Entries:            {total_entries}")
    print(f"Entries with Crops:       {entries_with_crops}")
    print(f"Faithful (tool but no crops): {faithful_entries}")
    print(f"No Tool Attempt:          {no_tool_entries}")
    print(f"Total Crops:              {total_crops}")

    if total_crops > 0:
        all_judgments = [j for r in output_records for j in r["crop_judgments"]]
        overall_crop_accuracy = sum(all_judgments) / len(all_judgments) * 100

        print()
        print("CROP ACCURACY:")
        print(f"  Overall:           {overall_crop_accuracy:.1f}% ({sum(all_judgments)}/{len(all_judgments)} contain object)")
        print(f"  All Crops Correct: {sum(1 for r in output_records if r['all_crops_correct'] and r['num_crops'] > 0)}/{entries_with_crops}")
        print(f"  Any Crop Correct:  {sum(1 for r in output_records if r['any_crop_correct'])}/{entries_with_crops}")

    # Faithful entries stats
    if faithful_entries > 0:
        faithful_with_answer = [r for r in output_records if r["faithful"] and r["answer_correct"] is not None]
        if faithful_with_answer:
            faithful_correct = sum(1 for r in faithful_with_answer if r["answer_correct"])
            faithful_accuracy = faithful_correct / len(faithful_with_answer) * 100
            print()
            print("FAITHFUL ENTRIES (Tool Attempted but No Crops):")
            print(f"  Total:             {faithful_entries}")
            print(f"  Answer Accuracy:   {faithful_accuracy:.1f}% ({faithful_correct}/{len(faithful_with_answer)} correct)")

    # Answer accuracy stats
    entries_with_answer = [r for r in output_records if r["answer_correct"] is not None]
    if entries_with_answer:
        answer_accuracy = sum(1 for r in entries_with_answer if r["answer_correct"]) / len(entries_with_answer) * 100
        print()
        print("ANSWER ACCURACY:")
        print(f"  Overall:           {answer_accuracy:.1f}% ({sum(1 for r in entries_with_answer if r['answer_correct'])}/{len(entries_with_answer)} correct)")

    # Print confusion matrix
    print_confusion_matrix(confusion_stats)

    # Copy sample images for inspection
    copy_sample_images(output_records, output_path, num_samples=30)

    print("\n" + "="*80)
    print(f"All results saved to: {output_dir}/")
    print(f"  - crop_accuracy_results.jsonl")
    print(f"  - crop_accuracy_confusion_matrix.json")
    print(f"  - {sum(1 for r in output_records[:30] if r['num_crops'] > 0)} sample directories (with original and crops)")
    print("="*80)

    return output_records


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_crop_accuracy.py <rollout_file> <result_xlsx> [output_dir]")
        print("\nExample:")
        print("  python analyze_crop_accuracy.py \\")
        print("    outputs/response/vllm-pixelreasoner/VStarBench_20251110041538.jsonl \\")
        print("    outputs/vllm-pixelreasoner/vllm-pixelreasoner_VStarBench_gpt-4o-mini_result.xlsx")
        print("\nOptional:")
        print("  python analyze_crop_accuracy.py \\")
        print("    outputs/response/vllm-pixelreasoner/VStarBench_20251110041538.jsonl \\")
        print("    outputs/vllm-pixelreasoner/vllm-pixelreasoner_VStarBench_gpt-4o-mini_result.xlsx \\")
        print("    custom_output_dir")
        sys.exit(1)

    rollout_file = sys.argv[1]
    result_xlsx = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None

    asyncio.run(main(rollout_file, result_xlsx, output_dir))
