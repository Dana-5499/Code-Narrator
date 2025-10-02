from google.colab import drive
drive.mount('/content/drive')

from datasets import load_from_disk

save_dir = "file:///content/drive/MyDrive/codesearchnet_processed"
ds = load_from_disk(save_dir)

print("train examples:", len(ds["train"]))

# Analyze and print the distribution of code languages
language_counts = ds["train"].to_pandas()["language"].value_counts()

print("\n=== Code Language Distribution ===")
print(language_counts)

!pip -q install -U "transformers>=4.44" "datasets>=2.20" accelerate
!pip -q uninstall -y bitsandbytes
!pip -q install -U bitsandbytes

# =================== Baseline loss on the test split - prior to any fine-tuning ===================
import math, torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import DataLoader

SAVE_DIR   = "/content/drive/MyDrive/codesearchnet_processed"
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
SPLIT      = "test"        # full test set
BATCH_SIZE = 2
MAX_LEN    = 1536  # maximum input value

# loading the test split out of the preprocessed dataset
dsd = load_from_disk(SAVE_DIR)
ds  = dsd[SPLIT]

# 2) Build chat-formatted text
SYS = ("You are a senior engineer. Summarize code for non-technical managers: "
       "plain English, 1–3 sentences, no jargon, no code, start with a verb.")
response_template = "<|im_start|>assistant\n"

def to_chat(ex):
    lang = (ex.get("language") or "code").lower()
    code = ex["code"]
    target = (ex.get("docstring") or "").strip()
    user = (f"Language: {lang}\n"
            f"Instruction: Explain clearly and concisely what this code does.\n"
            f"Code:\n```{lang}\n{code}\n```")
    text = (
        "<|im_start|>system\n" + SYS + "<|im_end|>\n" +
        "<|im_start|>user\n" + user + "<|im_end|>\n" +
        response_template + target + "<|im_end|>\n"
    )
    return {"text": text}

sft_ds = ds.map(to_chat, remove_columns=ds.column_names)

# 3) Tokenizer/model (4-bit to save memory)
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16
)
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token or tok.unk_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", quantization_config=bnb, trust_remote_code=True
).eval()

# 4) Tokenize + mask prompt tokens (labels = -100 before assistant)
def tokenize_and_mask(ex):
    full_text = ex["text"]
    assert response_template in full_text, "assistant tag not found"

    # prompt_only length in tokens (up to and including assistant tag)
    prefix, _ = full_text.split(response_template, 1)
    prompt_only = prefix + response_template
    prompt_ids = tok(prompt_only, truncation=True, max_length=MAX_LEN, add_special_tokens=True)["input_ids"]
    prompt_len = len(prompt_ids)

    # tokenize full text
    enc = tok(full_text, truncation=True, max_length=MAX_LEN, add_special_tokens=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # labels: copy then mask prompt tokens
    labels = input_ids.copy()
    cut = min(prompt_len, len(labels))
    for i in range(cut):
        labels[i] = -100  # ignored in CE

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

tok_ds = sft_ds.map(tokenize_and_mask, remove_columns=["text"])

# 5) Simple collate that pads input_ids/attention and labels correctly
def collate(batch):
    ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    ams = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]
    lbs = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]
    pad_id = tok.pad_token_id

    ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_id)
    ams = torch.nn.utils.rnn.pad_sequence(ams, batch_first=True, padding_value=0)
    lbs = torch.nn.utils.rnn.pad_sequence(lbs, batch_first=True, padding_value=-100)
    return {"input_ids": ids, "attention_mask": ams, "labels": lbs}

loader = DataLoader(tok_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

# 6) Compute CE only over non-masked positions (assistant target)
ce = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

@torch.no_grad()
def eval_loss():
    total_loss, total_items = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(model.device)
        attn      = batch["attention_mask"].to(model.device)
        labels    = batch["labels"].to(model.device)

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_tok = ce(shift_logits.view(-1, shift_logits.size(-1)),
                      shift_labels.view(-1)).view(shift_labels.size(0), -1)

        valid = (shift_labels != -100).float()
        loss_per_ex = (loss_tok * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-9)

        total_loss += float(loss_per_ex.sum())
        total_items += loss_per_ex.numel()

    avg_ce = total_loss / max(total_items, 1)
    ppl = math.exp(avg_ce) if avg_ce < 20 else float("inf")
    return avg_ce, ppl

avg_ce, ppl = eval_loss()
print(f"\nBASELINE (no fine-tune) — Test set ({len(tok_ds)} examples kept):")
print(f"  Avg cross-entropy: {avg_ce:.4f}")
print(f"  Perplexity:        {ppl:.2f}")


# =================== Baseline by language (no fine-tuning) ===================
import math, torch, collections
import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import DataLoader

SAVE_DIR   = "/content/drive/MyDrive/codesearchnet_processed"
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
SPLIT      = "test"        # full test set
BATCH_SIZE = 2
MAX_LEN    = 1536  # maximum input value

# ------------------ 1) Load dataset ------------------
dsd = load_from_disk(SAVE_DIR)
ds  = dsd[SPLIT]

# ------------------ 2) Chat-formatting (same as your setup) ------------------
SYS = ("You are a senior engineer. Summarize code for non-technical managers: "
       "plain English, 1–3 sentences, no jargon, no code, start with a verb.")
response_template = "<|im_start|>assistant\n"

def to_chat(ex):
    lang = (ex.get("language") or "code").lower()
    code = ex["code"]
    target = (ex.get("docstring") or "").strip()
    user = (f"Language: {lang}\n"
            f"Instruction: Explain clearly and concisely what this code does.\n"
            f"Code:\n```{lang}\n{code}\n```")
    text = (
        "<|im_start|>system\n" + SYS + "<|im_end|>\n" +
        "<|im_start|>user\n" + user + "<|im_end|>\n" +
        response_template + target + "<|im_end|>\n"
    )
    return {"text": text, "language_norm": lang}

# Keep original columns and add "text" + normalized language
with_text = ds.map(to_chat)

# ------------------ 3) Tokenizer/model (4-bit) ------------------
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16
)
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token or tok.unk_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", quantization_config=bnb, trust_remote_code=True
).eval()

# ------------------ 4) Tokenize + mask function (same logic) ------------------
def tokenize_and_mask_text(text: str):
    # prompt_only length in tokens (up to and including assistant tag)
    prefix, _ = text.split(response_template, 1)
    prompt_only = prefix + response_template
    prompt_ids = tok(prompt_only, truncation=True, max_length=MAX_LEN, add_special_tokens=True)["input_ids"]
    prompt_len = len(prompt_ids)

    # tokenize full text
    enc = tok(text, truncation=True, max_length=MAX_LEN, add_special_tokens=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # labels: copy then mask prompt tokens
    labels = input_ids.copy()
    cut = min(prompt_len, len(labels))
    for i in range(cut):
        labels[i] = -100  # ignored in CE

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def tokenize_and_mask(ex):
    out = tokenize_and_mask_text(ex["text"])
    return out

# ------------------ 5) Collate ------------------
def collate(batch):
    ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    ams = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]
    lbs = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]
    pad_id = tok.pad_token_id

    ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_id)
    ams = torch.nn.utils.rnn.pad_sequence(ams, batch_first=True, padding_value=0)
    lbs = torch.nn.utils.rnn.pad_sequence(lbs, batch_first=True, padding_value=-100)
    return {"input_ids": ids, "attention_mask": ams, "labels": lbs}

# ------------------ 6) Eval util (dataset -> avg CE, ppl) ------------------
ce = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

@torch.no_grad()
def eval_dataset(tokenized_dataset):
    loader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    total_loss, total_items = 0.0, 0

    for batch in loader:
        input_ids = batch["input_ids"].to(model.device)
        attn      = batch["attention_mask"].to(model.device)
        labels    = batch["labels"].to(model.device)

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_tok = ce(shift_logits.view(-1, shift_logits.size(-1)),
                      shift_labels.view(-1)).view(shift_labels.size(0), -1)

        valid = (shift_labels != -100).float()
        loss_per_ex = (loss_tok * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-9)

        total_loss += float(loss_per_ex.sum())
        total_items += loss_per_ex.numel()

    avg_ce = total_loss / max(total_items, 1)
    ppl = math.exp(avg_ce) if avg_ce < 20 else float("inf")
    return avg_ce, ppl

# ------------------ 7) Run per-language ------------------
# Collect unique languages (normalized); skip Nones/empties
langs = sorted({(x or "").lower() for x in with_text["language_norm"] if (x or "").strip()})

results = []
counts  = collections.Counter([l for l in with_text["language_norm"] if (l or "").strip()])

print(f"\nFound {len(langs)} languages in the test split.")
for lang in langs:
    # filter rows for this language
    lang_ds = with_text.filter(lambda ex: (ex.get("language_norm") or "") == lang)
    if len(lang_ds) == 0:
        continue

    # tokenize + mask for this slice
    lang_tok = lang_ds.map(tokenize_and_mask, remove_columns=[c for c in lang_ds.column_names if c != "text"])
    # Keep only the columns required by collate
    lang_tok.set_format(type="python", columns=["input_ids", "attention_mask", "labels"])

    avg_ce, ppl = eval_dataset(lang_tok)
    results.append({"language": lang, "count": len(lang_tok), "avg_ce": avg_ce, "perplexity": ppl})
    print(f"{lang:>12s} | n={len(lang_tok):5d} | CE={avg_ce:6.4f} | PPL={ppl:8.2f}")

# ------------------ 8) Tabular summary ------------------
df = pd.DataFrame(results).sort_values(["count", "avg_ce"], ascending=[False, True]).reset_index(drop=True)
print("\n=== Baseline by language (sorted by count desc, CE asc) ===")
print(df.to_string(index=False))

import matplotlib.pyplot as plt

# Sort by perplexity (ascending) for a nicer chart; show top 15 by count
top = df.sort_values("count", ascending=False).head(15).sort_values("perplexity")
plt.figure(figsize=(8, 5))
plt.barh(top["language"], top["perplexity"])
plt.xlabel("Perplexity (lower is better)")
plt.ylabel("Language")
plt.title("deepseek — Baseline Perplexity by Language (Test Split)")
plt.show()

# Create a bar chart for Average Cross-Entropy
plt.figure(figsize=(8, 5))
plt.barh(top["language"], top["avg_ce"])
plt.xlabel("Average Cross-Entropy (lower is better)")
plt.ylabel("Language")
plt.title("deepseek — Baseline Average Cross-Entropy by Language (Test Split)")
plt.show()

from datasets import load_from_disk
import os

py_dir = "/content/drive/MyDrive/codesearchnet_compact_by_lang/python"
assert os.path.exists(py_dir), f"Missing: {py_dir}"
ds_python = load_from_disk(py_dir)          # <-- DatasetDict (train/validation/test)

print("Splits:", list(ds_python.keys()))
print("train examples:", len(ds_python["train"]))
print("val examples:", len(ds_python["validation"]))
print("test examples:", len(ds_python["test"]))

# ====================== Tiny-slice overfit sanity check (Qwen2.5, Python-only, compact set) ======================
# 0) GPU sanity
!nvidia-smi -L || echo "No GPU found (Colab: Runtime -> Change runtime type -> GPU)"

# --- clean install set (stable for Colab + Qwen2.5 + TRL + bnb) ---
!pip -q uninstall -y transformers tokenizers datasets accelerate trl peft bitsandbytes wandb
!pip -q install --no-cache-dir \
  "transformers==4.44.2" "tokenizers==0.19.1" "datasets==2.20.0" \
  "accelerate==0.34.2" "trl==0.9.6" "peft==0.12.0" "bitsandbytes==0.47.0"

# ------------------ 1) Drive + imports ------------------
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, random, torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

os.environ["WANDB_DISABLED"] = "true"
torch.set_grad_enabled(True)  # ensure grad is on globally
random.seed(42)

# ------------------ 2) Load your *compact* Python dataset ------------------
py_dir = "/content/drive/MyDrive/codesearchnet_compact_by_lang/python"
assert os.path.exists(py_dir), f"Missing: {py_dir}"
ds_python = load_from_disk(py_dir)   # DatasetDict with train/validation/test

print("Splits:", list(ds_python.keys()))
print("Counts -> train:", len(ds_python["train"]),
      "| val:", len(ds_python["validation"]),
      "| test:", len(ds_python["test"]))

# ------------------ 3) Tiny slices (Python-only already) ------------------
TRAIN_N, EVAL_N = 16, 8

def good_doc(e):
    t = (e.get("docstring") or "").strip()
    return len(t) >= 10 and t[0].isalpha()

# quality filter, shuffle, then slice from the *compact* Python set
train_raw = ds_python["train"].filter(good_doc).shuffle(seed=42).select(range(min(TRAIN_N, len(ds_python["train"]))))
eval_raw  = ds_python["validation"].filter(good_doc).shuffle(seed=42).select(range(min(EVAL_N, len(ds_python["validation"]))))

SYS = ("You are a senior engineer. Summarize code for non-technical managers: "
       "plain English, 1–3 sentences, no jargon, no code, start with a verb.")

def to_chat(ex):
    lang = "python"
    code = ex["code"]
    target = (ex.get("docstring") or "").strip()
    user = (f"Language: {lang}\n"
            f"Instruction: Explain clearly and concisely what this code does.\n"
            f"Code:\n```{lang}\n{code}\n```")
    text = (
        "<|im_start|>system\n" + SYS + "<|im_end|>\n" +
        "<|im_start|>user\n" + user + "<|im_end|>\n" +
        "<|im_start|>assistant\n" + target + "<|im_end|>\n"
    )
    return {"text": text}

train_sft = train_raw.map(to_chat, remove_columns=train_raw.column_names)
eval_sft  =  eval_raw.map(to_chat,  remove_columns=eval_raw.column_names)

print(f"Tiny slices -> train: {len(train_sft)} | eval: {len(eval_sft)}")

# ------------------ 4) Tokenizer/model (4-bit) + explicit LoRA wrap ------------------
BASE = "deepseek-ai/deepseek-coder-1.3b-instruct"   # base coder (not instruct)
MAX_LEN = 1536

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token or tok.unk_token

model = AutoModelForCausalLM.from_pretrained(
    BASE, device_map="auto", quantization_config=bnb, trust_remote_code=True
)

# IMPORTANT: disable KV cache during training and prep for k-bit
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

# Make sure input embeddings get grad via hooks (sometimes needed explicitly)
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

# LoRA adapters
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)
model = get_peft_model(model, lora)

# Sanity: adapters trainable
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,}")

# ------------------ 5) Collator: mask everything before assistant ------------------
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tok,
    response_template=response_template,
)

# ------------------ 6) Trainer config ------------------
cfg = SFTConfig(
    output_dir="qwen2p5_overfit16_python_compact",
    num_train_epochs=30,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_seq_length=MAX_LEN,
    packing=False,
    bf16=False,
    fp16=True,
    logging_steps=5,
    save_steps=10_000,
    eval_strategy="no",        # focus purely on overfitting the 16
    report_to="none",
    dataset_text_field="text",
    optim="paged_adamw_32bit",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=train_sft,
    eval_dataset=None,  # overfit focus
    args=cfg,
    data_collator=collator,
)

print(f"CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ------------------ 7) Extra sanity checks ------------------
assert response_template in train_sft[0]["text"], "assistant tag missing in prompt"
assert any(p.requires_grad for n,p in model.named_parameters() if "lora_" in n), "LoRA adapters not trainable"

model.train()
tmp = tok(train_sft[0]["text"], return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)
out = model(**tmp, labels=tmp["input_ids"])
print("Loss requires_grad (should be True):", out.loss.requires_grad)  # should be True

# ------------------ 8) Train (loss should steadily drop toward ~0) ------------------
train_out = trainer.train()
print("\n=== TrainingOutput ===")
print(train_out)

# ------------------ 9) Show final step losses from the log history ------------------
hist = [h for h in trainer.state.log_history if "loss" in h]
print("\nLast 10 logged steps (loss should be very small):")
for row in hist[-10:]:
    pretty = {k: round(float(v), 6) if isinstance(v, (int, float)) else v for k,v in row.items()}
    print(pretty)

# ------------------ 10) Memorization check: generate on TRAIN items ------------------
@torch.no_grad()
def gen_on_train(n=8, max_new_tokens=120):
    subset = train_sft.select(range(min(n, len(train_sft))))
    outs = []

    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tok.eos_token_id] if tok.eos_token_id is not None else []
    if im_end_id is not None:
        eos_ids.append(im_end_id)

    for ex in subset:
        text = ex["text"]
        prompt_only, rest = text.split(response_template, 1)
        prompt_only = prompt_only + response_template

        inputs = tok(prompt_only, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)
        out = trainer.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_ids,
            pad_token_id=tok.pad_token_id,
        )
        new_tok_ids = out[0][inputs["input_ids"].shape[1]:]
        pred = tok.decode(new_tok_ids, skip_special_tokens=True).split("<|im_end|>")[0].strip()
        gt = rest.split("<|im_end|>")[0].strip()
        outs.append((pred, gt))
    return outs

pairs = gen_on_train(n=min(8, len(train_sft)))
print("\n=== Sample TRAIN predictions (should match targets closely) ===")
for i,(p,g) in enumerate(pairs, 1):
    print(f"\n[{i}]")
    print("GT  :", g[:400] + ("..." if len(g)>400 else ""))
    print("PRED:", p[:400] + ("..." if len(p)>400 else ""))

# ================== Plot training loss & learning rate (works after the overfit run) ==================
import os, math
import matplotlib.pyplot as plt

# 1) Grab logs from the trainer (must be in the same runtime)
logs = getattr(trainer.state, "log_history", [])
assert logs, "No logs found on trainer.state.log_history. Run training first."

loss_steps, loss_vals = [], []
lr_steps, lr_vals     = [], []
epoch_mark_steps      = []

running_step = 0
running_epoch = 0.0
for row in logs:
    step  = int(row.get("step", running_step))
    epoch = float(row.get("epoch", running_epoch))
    running_step = max(running_step, step)
    running_epoch = max(running_epoch, epoch)

    if "loss" in row:             # training loss (logged every logging_steps)
        loss_steps.append(step)
        loss_vals.append(float(row["loss"]))

    if "learning_rate" in row:
        lr_steps.append(step)
        lr_vals.append(float(row["learning_rate"]))

    # mark epoch boundaries if present
    if "epoch" in row and ("train_runtime" in row or "train_loss" in row or "eval_loss" in row):
        epoch_mark_steps.append(step)

print(f"Found {len(loss_vals)} train-loss points and {len(lr_vals)} LR points.")

# 2) Output folder on Drive
outdir = "/content/drive/MyDrive/llm_training_plots"
os.makedirs(outdir, exist_ok=True)

# 3) Plot: Training Loss vs Step
plt.figure(figsize=(7,4))
plt.plot(loss_steps, loss_vals, marker='o', markersize=3, linewidth=1)
for s in set(epoch_mark_steps):
    plt.axvline(s, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
plt.xlabel("Training step")
plt.ylabel("Training loss (cross-entropy)")
plt.title("Training Loss vs Step")
plt.grid(True, alpha=0.3)
loss_path = os.path.join(outdir, "overfit_loss_vs_step.png")
plt.tight_layout(); plt.savefig(loss_path, dpi=180); plt.show()

# 4) Plot: Learning Rate vs Step
plt.figure(figsize=(7,4))
plt.plot(lr_steps, lr_vals, linewidth=1)
for s in set(epoch_mark_steps):
    plt.axvline(s, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
plt.xlabel("Training step")
plt.ylabel("Learning rate")
plt.title("Learning Rate (cosine schedule) vs Step")
plt.grid(True, alpha=0.3)
lr_path = os.path.join(outdir, "overfit_lr_vs_step.png")
plt.tight_layout(); plt.savefig(lr_path, dpi=180); plt.show()

print("Saved plots:")
print(" -", loss_path)
print(" -", lr_path)

