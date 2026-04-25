import json
import sys
import torch
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# Check dependencies
try:
    import nltk
    nltk.download('punkt', quiet=True)
except ImportError:
    print("[ERROR] nltk not installed. Install with: pip install nltk")
    sys.exit(1)

try:
    from datasets import Dataset
except ImportError:
    print("[ERROR] datasets not installed. Install with: pip install datasets")
    sys.exit(1)

try:
    from nltk.tokenize import word_tokenize
except ImportError:
    print("[ERROR] nltk not installed. Install with: pip install nltk")
    sys.exit(1)

MODEL_NAME = "t5-small"

def format_semantic_input(entry):
    """Format an ap2d_simple_gt entry into T5 input/output pair"""
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
    
    output_text = entry["symbolic_explanation"]
    
    return {"input_text": input_text, "target_text": output_text}

def main():
    # Check for required input file
    try:
        with open("train_ap2d_simple_gt.json", "r") as f:
            train_data = json.load(f)
    except FileNotFoundError:
        print("[ERROR] train_ap2d_simple_gt.json not found. Run gt_gen.py on pose_2d/train_set/ first.")
        sys.exit(1)

    print(f"[INFO] Training samples: {len(train_data)}")

    # Format to input/output pairs
    train_hf_dataset = Dataset.from_list([
        format_semantic_input(entry) for entry in train_data
    ])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=256,
            truncation=True
        )
        labels = tokenizer(
            batch["target_text"],
            max_length=128,
            truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tokenized = train_hf_dataset.map(preprocess, batched=True, remove_columns=["input_text", "target_text"])

    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    args = TrainingArguments(
        output_dir="semantic_explainer_t5",
        per_device_train_batch_size=8,
        learning_rate=5e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="no",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=train_tokenized,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model("semantic_explainer_t5")
    tokenizer.save_pretrained("semantic_explainer_t5")

    print("[OK] Trained and saved semantic_explainer_t5")

if __name__ == "__main__":
    main()
