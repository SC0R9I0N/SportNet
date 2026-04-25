import json
import sys
import numpy as np
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

# Check dependencies
try:
    from rouge_score import rouge_scorer
except ImportError:
    print("[ERROR] rouge-score not installed. Install with: pip install rouge-score")
    sys.exit(1)

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except ImportError:
    print("[ERROR] nltk not installed. Install with: pip install nltk")
    sys.exit(1)

MODEL_DIR = "semantic_explainer_t5"

def format_semantic_input(entry):
    """Format an ap2d_simple_gt entry into T5 input"""
    action = entry["simple_action"]
    lk = entry["left_knee_angle"]
    rk = entry["right_knee_angle"]
    hip_norm = entry["hip_height_norm"]
    ankle_norm = entry["ankle_height_norm"]
    stride_norm = entry["stride_length_norm"]
    torso = entry["torso_angle"]
    arm_sym = entry["arm_symmetry"]
    leg_sym = entry["leg_symmetry"]
    
    input_text = (
        f"Action: {action}\n"
        f"Left knee angle: {lk:.2f}\n"
        f"Right knee angle: {rk:.2f}\n"
        f"Hip height: {hip_norm:.3f}\n"
        f"Ankle height: {ankle_norm:.3f}\n"
        f"Stride length: {stride_norm:.3f}\n"
        f"Torso angle: {torso:.2f}\n"
        f"Arm symmetry: {arm_sym:.3f}\n"
        f"Leg symmetry: {leg_sym:.3f}\n\n"
        f"Explanation:"
    )
    
    return input_text

def compute_bleu_score(reference, hypothesis):
    """Compute BLEU score (0-1 scale)"""
    try:
        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(hypothesis.lower())
        
        # Use smoothing function to handle short sentences
        smoothing = SmoothingFunction().method1
        bleu = sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing
        )
        return float(bleu)
    except:
        return 0.0

def compute_exact_match(reference, hypothesis):
    """Check if predictions exactly match ground truth"""
    return 1.0 if reference.strip() == hypothesis.strip() else 0.0

def compute_token_overlap(reference, hypothesis):
    """Compute token-level F1 score"""
    ref_tokens = set(word_tokenize(reference.lower()))
    hyp_tokens = set(word_tokenize(hypothesis.lower()))
    
    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return 0.0
    
    intersection = len(ref_tokens & hyp_tokens)
    precision = intersection / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
    recall = intersection / len(ref_tokens) if len(ref_tokens) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return float(f1)

def main():
    # Check for required model and files
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    except Exception as e:
        print(f"[ERROR] Failed to load model from {MODEL_DIR}: {e}")
        print("[INFO] Run train_semantic_explainer.py first to train the model.")
        sys.exit(1)
    
    # Check for required input file
    try:
        with open("ap2d_simple_gt.json", "r") as f:
            val_data = json.load(f)
    except FileNotFoundError:
        print("[ERROR] ap2d_simple_gt.json not found. Run gt_gen.py on pose_2d/valid_set/ first.")
        sys.exit(1)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Evaluating on {len(val_data)} validation samples")

    # ROUGE scorer
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL', 'rouge1', 'rouge2'], use_stemmer=True)

    predictions = []
    rouge_l_scores = []
    rouge_1_scores = []
    rouge_2_scores = []
    bleu_scores = []
    exact_matches = []
    token_f1_scores = []
    pred_lengths = []
    ref_lengths = []

    for i, entry in enumerate(val_data):
        ground_truth = entry["symbolic_explanation"]
        input_text = format_semantic_input(entry)

        # Generate prediction
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

        predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Compute multiple metrics
        rouge_scores = rouge_scorer_obj.score(ground_truth, predicted)
        rouge_l = rouge_scores['rougeL'].fmeasure
        rouge_1 = rouge_scores['rouge1'].fmeasure
        rouge_2 = rouge_scores['rouge2'].fmeasure
        
        bleu = compute_bleu_score(ground_truth, predicted)
        exact_match = compute_exact_match(ground_truth, predicted)
        token_f1 = compute_token_overlap(ground_truth, predicted)

        # Store metrics
        rouge_l_scores.append(rouge_l)
        rouge_1_scores.append(rouge_1)
        rouge_2_scores.append(rouge_2)
        bleu_scores.append(bleu)
        exact_matches.append(exact_match)
        token_f1_scores.append(token_f1)
        pred_lengths.append(len(predicted.split()))
        ref_lengths.append(len(ground_truth.split()))

        predictions.append({
            "input": input_text,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "rouge_l_score": float(rouge_l),
            "rouge_1_score": float(rouge_1),
            "rouge_2_score": float(rouge_2),
            "bleu_score": float(bleu),
            "exact_match": float(exact_match),
            "token_f1": float(token_f1),
        })

        if (i + 1) % 100 == 0:
            print(f"[INFO] Processed {i + 1}/{len(val_data)} samples")

    # Compute statistics
    avg_rouge_l = np.mean(rouge_l_scores)
    avg_rouge_1 = np.mean(rouge_1_scores)
    avg_rouge_2 = np.mean(rouge_2_scores)
    avg_bleu = np.mean(bleu_scores)
    avg_exact_match = np.mean(exact_matches)
    avg_token_f1 = np.mean(token_f1_scores)
    avg_pred_length = np.mean(pred_lengths)
    avg_ref_length = np.mean(ref_lengths)

    print(f"\n[RESULTS]")
    print(f"  === ROUGE Metrics ===")
    print(f"  Average ROUGE-L: {avg_rouge_l:.4f}")
    print(f"  Average ROUGE-1: {avg_rouge_1:.4f}")
    print(f"  Average ROUGE-2: {avg_rouge_2:.4f}")
    print(f"  === Other Metrics ===")
    print(f"  Average BLEU: {avg_bleu:.4f}")
    print(f"  Exact Match Rate: {avg_exact_match:.4f} ({int(avg_exact_match * len(val_data))}/{len(val_data)} exact matches)")
    print(f"  Average Token F1: {avg_token_f1:.4f}")
    print(f"  === Length Statistics ===")
    print(f"  Avg Prediction Length: {avg_pred_length:.1f} tokens")
    print(f"  Avg Reference Length: {avg_ref_length:.1f} tokens")
    print(f"  === Quality Thresholds ===")
    print(f"  Samples with ROUGE-L > 0.5: {sum(1 for s in rouge_l_scores if s > 0.5)}/{len(val_data)}")
    print(f"  Samples with ROUGE-L > 0.7: {sum(1 for s in rouge_l_scores if s > 0.7)}/{len(val_data)}")
    print(f"  Samples with BLEU > 0.5: {sum(1 for s in bleu_scores if s > 0.5)}/{len(val_data)}")
    print(f"  Samples with Token F1 > 0.6: {sum(1 for s in token_f1_scores if s > 0.6)}/{len(val_data)}")

    # Save results
    with open("semantic_explainer_results.json", "w") as f:
        json.dump({
            "statistics": {
                "rouge_l": {
                    "mean": float(avg_rouge_l),
                    "std": float(np.std(rouge_l_scores)),
                    "min": float(np.min(rouge_l_scores)),
                    "max": float(np.max(rouge_l_scores))
                },
                "rouge_1": {
                    "mean": float(avg_rouge_1),
                    "std": float(np.std(rouge_1_scores)),
                    "min": float(np.min(rouge_1_scores)),
                    "max": float(np.max(rouge_1_scores))
                },
                "rouge_2": {
                    "mean": float(avg_rouge_2),
                    "std": float(np.std(rouge_2_scores)),
                    "min": float(np.min(rouge_2_scores)),
                    "max": float(np.max(rouge_2_scores))
                },
                "bleu": {
                    "mean": float(avg_bleu),
                    "std": float(np.std(bleu_scores)),
                    "min": float(np.min(bleu_scores)),
                    "max": float(np.max(bleu_scores))
                },
                "exact_match_rate": float(avg_exact_match),
                "token_f1": {
                    "mean": float(avg_token_f1),
                    "std": float(np.std(token_f1_scores)),
                    "min": float(np.min(token_f1_scores)),
                    "max": float(np.max(token_f1_scores))
                },
                "length_stats": {
                    "avg_prediction_tokens": float(avg_pred_length),
                    "avg_reference_tokens": float(avg_ref_length),
                    "std_prediction_tokens": float(np.std(pred_lengths)),
                    "std_reference_tokens": float(np.std(ref_lengths))
                },
                "total_samples": len(val_data)
            },
            "predictions": predictions
        }, f, indent=2)

    print("[OK] Saved validation results to semantic_explainer_results.json")

if __name__ == "__main__":
    main()
