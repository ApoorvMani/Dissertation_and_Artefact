"""
Step 1 - download the victim model and the three reference tokenizers.
Run this once. Everything gets cached locally so later scripts load instantly.
"""

import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

sys.stdout.reconfigure(encoding="utf-8")

# the model we attack - DistilBERT toxicity classifier, WordPiece tokenizer
VICTIM = "martin-ha/toxic-comment-model"

# tokenizers we only use for measuring, never for classifying
REFERENCE_TOKENIZERS = {
    "xlnet": "xlnet-base-cased",        # SentencePiece Unigram
    "gpt2": "gpt2",                     # byte-level BPE
    "bert": "bert-base-uncased",        # WordPiece baseline
}


def main():
    # --- victim model + its own tokenizer ---
    print("downloading victim model...")
    victim_tok = AutoTokenizer.from_pretrained(VICTIM)
    victim_model = AutoModelForSequenceClassification.from_pretrained(VICTIM)
    victim_model.eval()  # inference only, we never train this

    # quick sanity check so we know it actually works
    sample = "you are such an idiot"
    inputs = victim_tok(sample, return_tensors="pt")
    with torch.no_grad():
        logits = victim_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]

    print("  label map:", victim_model.config.id2label)
    print("  test probs:", [round(p, 4) for p in probs.tolist()])
    print("  tokens:", victim_tok.tokenize(sample))

    # --- reference tokenizers ---
    for name, hf_id in REFERENCE_TOKENIZERS.items():
        print(f"downloading {name} tokenizer...")
        tok = AutoTokenizer.from_pretrained(hf_id)
        print(f"  {name} tokens:", tok.tokenize(sample))

    # --- check what device we have available ---
    print("\ncuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    print("\nstep 1 done - everything cached.")


if __name__ == "__main__":
    main()