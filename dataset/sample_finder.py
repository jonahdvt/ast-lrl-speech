import json
from typing import List, Dict, Any, Optional, Tuple
import evaluate
from pathlib import Path

# Configuration settings - Modify these values to control the script behavior
FIND_WORSE = True  # Set to True to find degradations instead of improvements
TOP_N = 15  # Number of top results to return
MIN_DIFF = 0.001  # Minimum absolute WER difference to include (0.05 = 5%)
LANGUAGE_CODES = [
    # "hi_in",
    # "pa_in",
    # "ta_in",
    # "te_in",
    # "ml_in",
    # "sw_ke", 
    # "ha_ng", 
    # "yo_ng",
    # "lg_ug", 
    # "ig_ng"
    "fr_fr",
]

wer_metric = evaluate.load("wer")

def load_transcriptions(
    file_path: str,
    id_key: str,
    transcription_key: str,
    gold_key: str = None
) -> Dict[str, Dict[str, Any]]:
    """Load and parse transcription data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        id_key: JSON field to use as the record ID
        transcription_key: JSON field for this model's hypothesis
        gold_key: JSON field for the reference (only for baseline)
    
    Returns:
        Dictionary mapping record IDs to their transcription data
    """
    # Check if file exists before attempting to open
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {file_path}")

    out = {}
    for rec in data:
        # Add error handling for missing keys
        if id_key not in rec:
            continue  # Skip records without ID key
            
        rec_id = rec[id_key]
        
        # Skip if transcription key is missing
        if transcription_key not in rec:
            continue
            
        entry = {"transcription": rec[transcription_key]}
        
        # Store under the exact same key
        if gold_key and gold_key in rec:
            entry[gold_key] = rec[gold_key]
        if "code" in rec:
            entry["code"] = rec["code"]
        out[rec_id] = entry

    return out

def calculate_wer(hypothesis: str, reference: str) -> float:
    """Calculate Word Error Rate between hypothesis and reference.
    
    Args:
        hypothesis: The transcription to evaluate
        reference: The gold standard transcription
        
    Returns:
        WER score (lower is better)
    """
    return wer_metric.compute(predictions=[hypothesis], references=[reference])

def find_transcription_comparisons(
    baseline_file: str,
    id_key: str,
    baseline_trans_key: str,
    gold_key: str,
    improved_specs: List[Dict[str, str]],
    top_n: int = 1,
    min_diff: float = 0.0,
    find_worse: bool = False
) -> List[Dict[str, Any]]:
    """Find the best improvements or worst degradations in transcription quality across models.
    
    Args:
        baseline_file: Path to baseline results file
        id_key: JSON field to use as the record ID
        baseline_trans_key: JSON field for baseline transcription
        gold_key: JSON field for the reference transcription
        improved_specs: List of dictionaries specifying improved models
        top_n: Number of top results to return per model
        min_diff: Minimum absolute WER difference to include in results
        find_worse: If True, find cases where models performed worse than baseline
    
    Returns:
        List of dictionaries containing comparison details
    """
    # Load baseline (with gold_key)
    try:
        baseline_map = load_transcriptions(
            baseline_file, id_key, baseline_trans_key, gold_key
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading baseline file: {e}")
        return []

    # Dictionary to store results by file_id
    results_by_file = {}
    
    for spec in improved_specs:
        try:
            imp_map = load_transcriptions(
                spec["file_path"], id_key, spec["transcription_key"]
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading comparison model file {spec['file_path']}: {e}")
            continue
            
        model_name = spec.get("name", spec["file_path"])

        for rec_id, base_rec in baseline_map.items():
            # Skip if the comparison model has no entry
            if rec_id not in imp_map:
                continue

            # Read it under the same gold_key
            gold = base_rec.get(gold_key)
            if gold is None:
                print(f"Warning: Record {rec_id!r} has no {gold_key} in baseline")
                continue
                
            base_hyp = base_rec["transcription"]
            imp_hyp = imp_map[rec_id]["transcription"]

            wer_base = calculate_wer(base_hyp, gold)
            wer_imp = calculate_wer(imp_hyp, gold)
            diff = wer_base - wer_imp  # Positive means improvement, negative means worse
            
            # Filter based on find_worse flag
            if find_worse and diff >= 0:
                continue  # Skip improvements when looking for degradations
            if not find_worse and diff <= 0:
                continue  # Skip degradations when looking for improvements
                
            # Check if difference exceeds threshold (use absolute value)
            if abs(diff) < min_diff:
                continue
                
            # Create or update the file entry
            if rec_id not in results_by_file:
                results_by_file[rec_id] = {
                    "file_id": rec_id,
                    "code": base_rec.get("code"),
                    "gold": gold,
                    "baseline": base_hyp,
                    "wer_base": wer_base * 100,  # Store as percentage
                    "models": []
                }
            
            # Add this model's results
            results_by_file[rec_id]["models"].append({
                "name": model_name,
                "transcription": imp_hyp,
                "wer": wer_imp * 100,  # Store as percentage
                "wer_diff": diff * 100  # Store as percentage (positive=better, negative=worse)
            })

    # Convert dictionary to list
    all_results = list(results_by_file.values())
    
    # Sort models within each file by WER difference magnitude (considering direction)
    for result in all_results:
        if find_worse:
            # For worse cases, sort by most negative (worst) first
            result["models"].sort(key=lambda x: x["wer_diff"])
        else:
            # For improvements, sort by most positive (best) first
            result["models"].sort(key=lambda x: x["wer_diff"], reverse=True)
    
    # Sort files by the most extreme difference from any model
    if find_worse:
        # Sort by worst degradation for find_worse mode
        all_results.sort(key=lambda x: min(model["wer_diff"] for model in x["models"]) if x["models"] else 0)
    else:
        # Sort by best improvement for improvement mode
        all_results.sort(key=lambda x: max(model["wer_diff"] for model in x["models"]) if x["models"] else 0, reverse=True)
    
    # Limit to top_n files
    return all_results[:top_n]

def format_consolidated_result(result: Dict[str, Any], find_worse: bool = False) -> str:
    """Format a consolidated result for printing.
    
    Args:
        result: Dictionary containing consolidated result data
        find_worse: Whether this is for degradation results
        
    Returns:
        Formatted string representation
    """
    output = []
    output.append(f"File: {result['file_id']} (code {result['code']})")
    output.append(f"Baseline WER: {result['wer_base']:.2f}%")
    output.append(f"Gold: {result['gold']}")
    output.append(f"Baseline: {result['baseline']}")
    
    if find_worse:
        output.append("\nModel degradations:")
    else:
        output.append("\nModel improvements:")
        
    for i, model in enumerate(result["models"], 1):
        direction = "worse" if model["wer_diff"] < 0 else "better"
        output.append(f"  {i}. {model['name']}: WER {model['wer']:.2f}% ({abs(model['wer_diff']):.2f}% {direction})")
        output.append(f"     {model['transcription']}")
    
    return "\n".join(output)

def main():
    """Main entry point for the script."""
    # Use configuration settings from top of file
    find_worse = FIND_WORSE
    top_n = TOP_N
    min_diff = MIN_DIFF
    codes = LANGUAGE_CODES
    
    # Ensure output directory exists
    result_type = "WORSE_RESULTS" if find_worse else "SAMPLE_RESULTS"
    output_dir = Path(result_type)
    output_dir.mkdir(exist_ok=True)
    
    for code in codes:
        print(f"\nProcessing language code: {code}")
        
        baseline_file = f"RESULTS/whisper_l_baseline/{code}.json"
        id_key = "file_name"
        baseline_trans_key = "whisper_l"
        gold_key = "gold_transcript"

        improved_specs = [
            {
                "file_path": f"RESULTS/whisper_l_uni_results/{code}.json",
                "transcription_key": "whisper_l_ft",  # Note the key name difference
                "name": "whisper_l_uni"
            },
            {
                "file_path": f"RESULTS/whisper_l_plus_results/{code}.json",
                "transcription_key": "whisper_l_ft",  # Note the key name difference
                "name": "whisper_l_plus"
            },
            {
                "file_path": f"RESULTS/ft_whisper_results/l_ft_whisper_results/indic-5/{code}_indic.json",
                "transcription_key": "whisper_l_ft",
                "name": "whisper_l_multi"
            }
        ]

        results = find_transcription_comparisons(
            baseline_file,
            id_key,
            baseline_trans_key,
            gold_key,
            improved_specs,
            top_n=top_n,
            min_diff=min_diff,
            find_worse=find_worse
        )

        mode_description = "degradations" if find_worse else "improvements"
        print(f"Found {len(results)} top {mode_description} across multiple models.")
        
        # Save results to output JSON file
        output_file = output_dir / f"{code}.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results to {output_file}: {e}")
        
        # Print sample results to console
        sample_count = min(2, len(results))
        if sample_count > 0:
            print(f"\nShowing top {sample_count} {mode_description}:")
            for r in results[:sample_count]:
                print(format_consolidated_result(r, find_worse))
                print()  # Extra newline for readability

if __name__ == "__main__":
    main()