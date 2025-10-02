# ===== Cell 1: Mount + imports + dataset load =====
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, random, math, gc, torch
from datasets import load_from_disk

# Reproducibility setup
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Path to the compact Python split of CommitPackFT saved on Google Drive.
# NOTE: this script expects the dataset to already be prepared on Drive.
commit_dir = "/content/drive/MyDrive/commitpackft_compact_by_lang/python"
assert os.path.exists(commit_dir), f"Missing dataset at: {commit_dir}"
ds_commit = load_from_disk(commit_dir)

print("Splits:", list(ds_commit.keys()))
print({k: len(ds_commit[k]) for k in ds_commit.keys()})

# Disable W&B logging and make CUDA memory allocation more stable in Colab
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ===== Cell 2: Prepare training/eval texts =====
# Keep only commit rows that have a non-trivial, natural-language target
def good_commit(ex):
    tgt = (ex.get("target") or "").strip()
    return len(tgt) >= 10 and tgt[0].isalpha()

# Filter raw splits
train_raw = ds_commit["train"].filter(good_commit)
val_raw   = ds_commit["validation"].filter(good_commit)
test_raw  = ds_commit["test"].filter(good_commit)
print("Prepared filter sizes:", len(train_raw), len(val_raw), len(test_raw))

# System instruction used for the chat template
SYS = ("You are a senior engineer. Summarize commit changes for non-technical managers: "
       "plain English, 1–3 sentences, no jargon, no code, start with a verb.")

# Convert commit rows into ChatML-style text with system/user/assistant blocks
def to_chat_commit(ex):
    before = (ex.get("old_code") or ex.get("old_contents") or "").strip()
    after  = (ex.get("new_code") or ex.get("new_contents") or "").strip()
    target = (ex.get("target") or "").strip()
    user = (f"Language: python\nInstruction: Explain clearly what changed.\n\n"
            f"BEFORE:\n```python\n{before}\n```\n\nAFTER:\n```python\n{after}\n```")
    return {"text":
        "<|im_start|>system\n" + SYS + "<|im_end|>\n" +
        "<|im_start|>user\n" + user + "<|im_end|>\n" +
        "<|im_start|>assistant\n" + target + "<|im_end|>\n"
    }

# Map raw datasets into SFT-ready chat text (keep only 'text' column)
train_sft = train_raw.map(to_chat_commit, remove_columns=train_raw.column_names)
val_sft   = val_raw.map(to_chat_commit,   remove_columns=val_raw.column_names)
test_sft  = test_raw.map(to_chat_commit,  remove_columns=test_raw.column_names)

print({"train": len(train_sft), "val": len(val_sft), "test": len(test_sft)})

# ===== Cell 4: Model loader + LoRA wrapper =====
# Loads model/tokenizer in 4-bit (bitsandbytes) and wraps with LoRA adapters
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

HF_TOKEN = None  # paste an HF token if you use gated repos; otherwise keep None

def load_model_and_tokenizer(model_name: str, hf_token=None):
    # 4-bit quantization config (NF4) to fit 1.3B on Colab GPU memory
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16
    )
    tok = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True, token=hf_token
    )
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token or tok.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", quantization_config=bnb,
        trust_remote_code=True, token=hf_token
    )
    model.config.use_cache = False  # avoids CUDA graph issues while training
    return tok, model

def lora_wrap(model, use_gc=False):
    # Prepare for k-bit training and enable gradient checkpointing (optional)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gc)
    if hasattr(model, "gradient_checkpointing_enable") and use_gc:
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    # LoRA config: which projection matrices get adapters, and their rank/alpha/dropout
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    return get_peft_model(model, lcfg)

# ===== Cell X: Model + LoRA helpers (drop-in) =====
# NOTE: This section duplicates the helpers above with tiny differences
# (e.g., padding_side="right" and use_gc True by default). It’s functionally equivalent.
import gc, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

HF_TOKEN = None  # set if you need a gated model; otherwise keep None

def load_model_and_tokenizer(model_name: str, hf_token=None):
    """Loads tokenizer + 4-bit quantized model (NF4) and disables cache for training."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True, token=hf_token
    )
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    tok.padding_side = "right"  # safer for fp16 training in TRL

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb,
        trust_remote_code=True,
        token=hf_token,
    )
    model.config.use_cache = False
    return tok, model

def lora_wrap(model, use_gc=True):
    """Preps 4-bit model for LoRA training and applies LoRA to common proj modules."""
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gc)
    if hasattr(model, "gradient_checkpointing_enable") and use_gc:
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    return get_peft_model(model, lcfg)

def free_cuda():
    # Utility to nudge the Colab allocator to free memory between runs
    try:
        torch.cuda.empty_cache()
        gc.collect()
    except:
        pass


# Build CommitPackFT train_sft / val_sft / test_sft 
from datasets import load_from_disk, DatasetDict
import os, re

# Disk path again (redundant but harmless if you run cells out of order)
COMMIT_DIR = "/content/drive/MyDrive/commitpackft_compact_by_lang/python"
assert os.path.exists(COMMIT_DIR), f"Missing commit dataset at: {COMMIT_DIR}"
ds_commit: DatasetDict = load_from_disk(COMMIT_DIR)

# ---- helpers ----
# If target is missing, synthesize from subject/message to avoid dropping rows
def _synth_target(ex):
    subj = (ex.get("subject") or "").strip()
    msg  = (ex.get("message") or "").strip()
    if msg and msg.lower() != subj.lower():
        t = (subj + " " + msg).strip() if subj else msg
    else:
        t = subj or msg
    return t

# Basic quality filter: needs before/after code and a reasonable explanation
def has_min_fields(ex):
    before = (ex.get("old_code") or ex.get("old_contents") or "").strip()
    after  = (ex.get("new_code") or ex.get("new_contents") or "").strip()
    tgt    = (ex.get("target") or _synth_target(ex) or "").strip()
    return bool(before) and bool(after) and len(tgt) >= 10

# System prompt for this chat template
SYS = ("You are a senior engineer. Explain the functional change between two code revisions "
       "in plain English, 1–3 sentences, no jargon, no code, start with a verb.")

ASSIST_TAG = "<|im_start|>assistant\n"  # used later for masking
def to_chat_commit(ex):
    # Convert a CommitPack example into supervised chat format
    before = (ex.get("old_code") or ex.get("old_contents") or "").strip()
    after  = (ex.get("new_code") or ex.get("new_contents") or "").strip()
    target = (ex.get("target") or _synth_target(ex) or "").strip()
    user = (
        "Language: python\n"
        "Instruction: Explain clearly and concisely what changed between BEFORE and AFTER.\n"
        f"BEFORE:\n```python\n{before}\n```\n\nAFTER:\n```python\n{after}\n```"
    )
    return {"text":
        "<|im_start|>system\n" + SYS + "<|im_end|>\n" +
        "<|im_start|>user\n"   + user + "<|im_end|>\n" +
        ASSIST_TAG + target + "<|im_end|>\n"
    }

# ---- filter + map each split ----
train_raw = ds_commit["train"].filter(has_min_fields)
val_raw   = ds_commit.get("validation", ds_commit["train"].select(range(0)))  # leave empty if missing
test_raw  = ds_commit["test"].filter(has_min_fields) if "test" in ds_commit else None

train_sft = train_raw.map(to_chat_commit, remove_columns=train_raw.column_names)
val_sft   = val_raw.map(to_chat_commit,   remove_columns=val_raw.column_names) if len(val_raw) else None
test_sft  = test_raw.map(to_chat_commit,  remove_columns=test_raw.column_names) if test_raw is not None else None

print("Prepared chat sizes:",
      {"train": len(train_sft),
       "val": (len(val_sft) if val_sft is not None else 0),
       "test": (len(test_sft) if test_sft is not None else 0)})

# Make ASSIST_TAG available to other cells (used by the collator/masking logic)
response_template = ASSIST_TAG

# ===== Cell 5: Train on Commit TRAIN; Eval on TEST =====
# TRL SFTTrainer setup: trains on Commit train set, evaluates every N steps on Commit test.
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

# Chosen base model + hyperparameters (ported from your CodeSearchNet sweeps)
MODEL_TARGET    = "deepseek-ai/deepseek-coder-1.3b-instruct"
LR_FIXED        = 2e-4
SCHEDULER       = "constant_with_warmup"
WARMUP_RATIO    = 0.05
NUM_EPOCHS      = 1
BATCH_SIZE      = 4
GRAD_ACC        = 4
MAX_SEQ_LENGTH  = 1536
USE_GC          = True
EVAL_STEPS      = 100
LOGGING_STEPS   = 10
OPTIMIZER_NAME  = "adafactor"

print(f"\n=== Train on Commit TRAIN | Eval on TEST | lr={LR_FIXED} | bsz={BATCH_SIZE} ===")
tok, base_model = load_model_and_tokenizer(MODEL_TARGET)
tok.padding_side = "right"
model = lora_wrap(base_model, use_gc=USE_GC)

# Collator that masks the prompt and keeps only assistant tokens as labels
collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tok, response_template="<|im_start|>assistant\n"
)

# Trainer configuration
cfg = SFTConfig(
    output_dir=f"ft_commit_train_evaltest_{OPTIMIZER_NAME}",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR_FIXED,
    lr_scheduler_type=SCHEDULER,
    warmup_ratio=WARMUP_RATIO,
    max_seq_length=MAX_SEQ_LENGTH,
    bf16=False, fp16=True,
    logging_steps=LOGGING_STEPS,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=10_000,        # effectively disables periodic checkpointing to save space
    report_to="none",
    dataset_text_field="text",
    optim=OPTIMIZER_NAME,
)

# Train on train_sft, evaluate on test_sft every eval_steps
trainer = SFTTrainer(
    model=model, tokenizer=tok,
    train_dataset=train_sft, eval_dataset=test_sft,
    args=cfg, data_collator=collator,
)

trainer.train()

# ---- Plot eval loss vs steps (from trainer logs) ----
def _extract_eval(trainer):
    logs = getattr(trainer.state, "log_history", [])
    es, ev = [], []
    for row in logs:
        if "eval_loss" in row:
            es.append(int(row["step"])); ev.append(float(row["eval_loss"]))
    return es, ev

es, ev = _extract_eval(trainer)
plt.figure(figsize=(10,5))
if es: plt.plot(es, ev, linewidth=2, label="Commit TEST CE")
plt.title("Eval loss vs steps (Commit TEST)")
plt.xlabel("Step"); plt.ylabel("Eval loss")
plt.grid(True, alpha=.3); plt.legend(); plt.show()

import matplotlib.pyplot as plt

# Extract raw per-step training loss from trainer logs
def extract_train_loss(trainer):
    logs = getattr(trainer.state, "log_history", [])
    steps, losses = [], []
    for row in logs:
        if "loss" in row:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    return steps, losses

steps, losses = extract_train_loss(trainer)

plt.figure(figsize=(8,4))
plt.plot(steps, losses, label="Train loss", color="tab:blue")
plt.xlabel("Step"); plt.ylabel("Loss")
plt.title("Training loss vs steps (Commit dataset)")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
plt.show()

# === Minimal helpers to evaluate on Commit TEST ===
# Compute token-level Cross-Entropy and PPL on the test set using assistant-only labels.
import torch, math
from torch.utils.data import DataLoader

response_template = "<|im_start|>assistant\n"
MAX_LEN_DEFAULT = 1536
ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

# Tokenize + mask everything before assistant tag
def tokenize_and_mask_factory(tok, max_len=MAX_LEN_DEFAULT):
    def _f(ex):
        full_text = ex["text"]
        prefix, _ = full_text.split(response_template, 1)
        prompt_only = prefix + response_template

        prompt_ids = tok(prompt_only, truncation=True, max_length=max_len, add_special_tokens=True)["input_ids"]
        prompt_len = len(prompt_ids)

        enc = tok(full_text, truncation=True, max_length=max_len, add_special_tokens=True)
        input_ids = enc["input_ids"]; attention_mask = enc["attention_mask"]

        labels = input_ids.copy()
        cut = min(prompt_len, len(labels))
        for i in range(cut): labels[i] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return _f

# Pad variable-length sequences into batch tensors
def pad_collate(tok):
    def _c(batch):
        ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        ams = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]
        lbs = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]
        pad_id = tok.pad_token_id
        ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_id)
        ams = torch.nn.utils.rnn.pad_sequence(ams, batch_first=True, padding_value=0)
        lbs = torch.nn.utils.rnn.pad_sequence(lbs, batch_first=True, padding_value=-100)
        return {"input_ids": ids, "attention_mask": ams, "labels": lbs}
    return _c

# Evaluate CE/PPL across the whole loader
@torch.no_grad()
def eval_ce_ppl(model, loader):
    model.eval()
    total_loss, total_items = 0.0, 0
    for batch in loader:
        ids = batch["input_ids"].to(model.device)
        ams = batch["attention_mask"].to(model.device)
        lbs = batch["labels"].to(model.device)

        logits = model(input_ids=ids, attention_mask=ams).logits
        sl, tl = logits[:, :-1, :].contiguous(), lbs[:, 1:].contiguous()

        loss_tok = ce_loss(sl.view(-1, sl.size(-1)), tl.view(-1)).view(tl.size(0), -1)
        valid = (tl != -100).float()
        loss_per_ex = (loss_tok * valid).sum(1) / (valid.sum(1) + 1e-9)

        total_loss += float(loss_per_ex.sum())
        total_items += loss_per_ex.numel()
    ce = total_loss / max(total_items, 1)
    ppl = math.exp(ce) if ce < 20 else float("inf")
    return ce, ppl

# Prepare test dataloader (tokenized & masked)
tnm = tokenize_and_mask_factory(tok)
test_tok = test_sft.map(tnm, remove_columns=["text"])
coll = pad_collate(tok)
test_loader = DataLoader(test_tok, batch_size=2, shuffle=False, collate_fn=coll)

# Final Commit TEST metrics (CE and PPL)
test_ce, test_ppl = eval_ce_ppl(trainer.model, test_loader)
print(f"Commit TEST — CE: {test_ce:.4f} | PPL: {test_ppl:.2f}")
