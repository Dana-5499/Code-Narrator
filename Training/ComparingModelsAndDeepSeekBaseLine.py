# ===================== Cell 2: Verify environment =====================
# Purpose: Print versions of core libs and confirm CUDA + bitsandbytes are available.
# Tip: For Colab stability, pin versions (e.g., numpy 1.26.x, transformers 4.44.x, trl 0.9.6).
# If bitsandbytes import fails, you'll still run in CPU/FP16 mode without 4-bit quantization.
import numpy, torch, transformers, datasets, trl, peft
print("NumPy:", numpy.__version__)
print("Transformers:", transformers.__version__)
print("Datasets:", datasets.__version__)
print("TRL:", trl.__version__)
print("PEFT:", peft.__version__)
try:
    import bitsandbytes as bnb
    print("bitsandbytes:", bnb.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("bitsandbytes import failed:", repr(e))

# ===================== Cell 3: Mount + Dataset prep =====================
# Purpose: Mount Drive, load your compact Python-only CodeSearchNet subset, quality-filter docstrings,
# and format examples as chat turns using a simple, consistent special-token template.
# Notes:
# - MODEL_NAMES contains both base and instruct variants; we’ll baseline and optionally fine-tune all.
# - The chat format uses <|im_start|>/... tags; masking later depends on "<|im_start|>assistant\n".
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, random, torch, math
from datasets import load_from_disk

RUN_BASELINE = True     # Evaluate zero-shot CE/PPL for each model
RUN_FINETUNE = True     # Do LoRA fine-tuning runs on the Python subset
NUM_EPOCHS   = 1        # Keep 1 epoch for quick sweep; adjust for quality vs time
SEED         = 42
random.seed(SEED)

HF_TOKEN = None  # Only set if you need gated/private models

MODEL_NAMES = [
    "Qwen/Qwen2.5-Coder-1.5B",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "01-ai/Yi-Coder-1.5B",
    # Added:
    "deepseek-ai/deepseek-coder-1.3b-instruct",  # ~1.3B code model (Instruct)
]

# Load the compact Python dataset saved earlier (train/validation/test)
py_dir = "/content/drive/MyDrive/codesearchnet_compact_by_lang/python"
assert os.path.exists(py_dir), f"Missing: {py_dir}"
ds_python = load_from_disk(py_dir)
print("Splits:", list(ds_python.keys()))
print({k: len(ds_python[k]) for k in ds_python.keys()})

# Filter out short/odd docstrings; basic signal-quality gate
def good_doc(ex):
    t = (ex.get("docstring") or "").strip()
    return len(t) >= 10 and t[0].isalpha()

train_raw = ds_python["train"].filter(good_doc)
val_raw   = ds_python["validation"].filter(good_doc)
test_raw  = ds_python["test"].filter(good_doc)

# Chat template (consistent across models; some models have their own templates—this generic one works fine for CE)
SYS = ("You are a senior engineer. Summarize code for non-technical managers: "
       "plain English, 1–3 sentences, no jargon, no code, start with a verb.")
def to_chat(ex):
    code = ex["code"]
    target = (ex.get("docstring") or "").strip()
    user = (f"Language: python\nInstruction: Explain clearly what this code does.\n"
            f"Code:\n```python\n{code}\n```")
    return {"text":
        "<|im_start|>system\n" + SYS + "<|im_end|>\n" +
        "<|im_start|>user\n" + user + "<|im_end|>\n" +
        "<|im_start|>assistant\n" + target + "<|im_end|>\n"
    }

# Map to chat strings; keep only the "text" column to simplify training
train_sft = train_raw.map(to_chat, remove_columns=train_raw.column_names)
val_sft   = val_raw.map(to_chat,   remove_columns=val_raw.column_names)
test_sft  = test_raw.map(to_chat,  remove_columns=test_raw.column_names)
print("Prepared:", {"train": len(train_sft), "val": len(val_sft), "test": len(test_sft)})

# ===================== Cell 4: Helpers =====================
# Purpose: Tokenize with assistant-only loss masking; pad batches; compute CE/PPL.
# Key detail: We mask tokens up to (and including) the assistant tag so loss is only on the assistant span.
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MAX_LEN = 1536
response_template = "<|im_start|>assistant\n"         # anchor used for prompt/answer split
ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

def tokenize_and_mask_factory(tok):
    # Returns a function that makes input_ids/attention_mask and labels with -100 mask for prompt tokens
    def _f(ex):
        full_text = ex["text"]
        prefix, _ = full_text.split(response_template, 1)
        prompt_only = prefix + response_template
        prompt_ids = tok(prompt_only, truncation=True, max_length=MAX_LEN, add_special_tokens=True)["input_ids"]
        prompt_len = len(prompt_ids)
        enc = tok(full_text, truncation=True, max_length=MAX_LEN, add_special_tokens=True)
        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
        labels = input_ids.copy()
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return _f

def pad_collate(tok):
    # Dynamic pad to the longest in batch; labels padded with -100; attention with 0
    def _c(batch):
        ids = [torch.tensor(x["input_ids"]) for x in batch]
        ams = [torch.tensor(x["attention_mask"]) for x in batch]
        lbs = [torch.tensor(x["labels"]) for x in batch]
        pad_id = tok.pad_token_id
        ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_id)
        ams = torch.nn.utils.rnn.pad_sequence(ams, batch_first=True, padding_value=0)
        lbs = torch.nn.utils.rnn.pad_sequence(lbs, batch_first=True, padding_value=-100)
        return {"input_ids": ids, "attention_mask": ams, "labels": lbs}
    return _c

@torch.no_grad()
def eval_ce_ppl(model, loader):
    # Computes token-avg CE (assistant-only) and PPL; safe for larger batches with no_grad.
    model.eval()
    total_loss, total_items = 0.0, 0
    for batch in loader:
        ids, attn, lbs = batch["input_ids"].to(model.device), batch["attention_mask"].to(model.device), batch["labels"].to(model.device)
        logits = model(input_ids=ids, attention_mask=attn).logits
        shift_logits, shift_labels = logits[:, :-1, :], lbs[:, 1:]
        loss_tok = ce_loss(shift_logits.reshape(-1, shift_logits.size(-1)),
                           shift_labels.reshape(-1)).view(shift_labels.size(0), -1)
        valid = (shift_labels != -100).float()
        loss_per_ex = (loss_tok * valid).sum(1) / (valid.sum(1) + 1e-9)
        total_loss += float(loss_per_ex.sum()); total_items += loss_per_ex.numel()
    avg_ce = total_loss / max(total_items, 1)
    ppl = math.exp(avg_ce) if avg_ce < 20 else float("inf")
    return avg_ce, ppl

# ===================== Cell 5: Loader + LoRA =====================
# Purpose: Load models in 4-bit NF4 (if bitsandbytes available) and wrap with LoRA adapters for PEFT training.
# Gotcha: Always set a pad_token (fallback to eos/unk) to avoid training-time errors with packed sequences.
def load_model_and_tokenizer(model_name: str, hf_token=None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16
    )
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", quantization_config=bnb,
        trust_remote_code=True, token=hf_token
    )
    model.config.use_cache = False   # disable cache during training for gradient checkpointing compat
    return tok, model

def lora_wrap(model):
    # Prepares k-bit model for LoRA; attaches adapters to attention/MLP projection modules.
    model = prepare_model_for_kbit_training(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    return get_peft_model(model, lcfg)

# ===================== Cell 6: Baseline Eval =====================
# Purpose: Zero-shot CE/PPL on VAL/TEST for each base model. Helpful to select a strong starting point.
# Tip: Clear VRAM between models if you hit OOM; batch_size=2 is conservative for 1–1.5B in 4-bit.
from torch.utils.data import DataLoader

if RUN_BASELINE:
    results = {}
    for name in MODEL_NAMES:
        print(f"\n=== Baseline eval: {name} ===")
        tok, model = load_model_and_tokenizer(name, hf_token=HF_TOKEN)
        tnm = tokenize_and_mask_factory(tok)
        val_tok  = val_sft.map(tnm, remove_columns=["text"])
        test_tok = test_sft.map(tnm, remove_columns=["text"])
        coll = pad_collate(tok)
        val_loader  = DataLoader(val_tok,  batch_size=2, shuffle=False, collate_fn=coll)
        test_loader = DataLoader(test_tok, batch_size=2, shuffle=False, collate_fn=coll)
        val_ce, val_ppl = eval_ce_ppl(model, val_loader)
        test_ce, test_ppl = eval_ce_ppl(model, test_loader)
        print(f"Validation — CE: {val_ce:.4f} | PPL: {val_ppl:.2f}")
        print(f"Test       — CE: {test_ce:.4f} | PPL: {test_ppl:.2f}")
        results[name] = dict(val_ce=val_ce, val_ppl=val_ppl,
                             test_ce=test_ce, test_ppl=test_ppl)
    print("\n(Baseline summary) ->", results)

# ===================== Cell 7: Fine-Tuning =====================
# Purpose: Run LoRA SFT on the training split for each model; eval during training on VAL.
# Knobs: change per_device_train_batch_size/grad_acc/num_train_epochs for quality vs. time/VRAM.
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

FT_RUNS = {}
if RUN_FINETUNE:
    for name in MODEL_NAMES:
        print(f"\n=== Fine-tune: {name} ===")
        tok, base_model = load_model_and_tokenizer(name)
        model = lora_wrap(base_model)
        collator = DataCollatorForCompletionOnlyLM(tokenizer=tok, response_template=response_template)
        cfg = SFTConfig(
            output_dir=f"ft_{os.path.basename(name).lower()}_python",
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            max_seq_length=MAX_LEN,
            packing=False,
            bf16=False, fp16=True,
            logging_steps=10,
            evaluation_strategy="steps", eval_steps=100,
            save_steps=5_000, report_to="none",
            dataset_text_field="text", optim="paged_adamw_32bit",
        )
        trainer = SFTTrainer(
            model=model, tokenizer=tok,
            train_dataset=train_sft, eval_dataset=val_sft,
            args=cfg, data_collator=collator,
        )
        trainer.train()
        FT_RUNS[name] = dict(trainer=trainer, tok=tok)

# ===================== Cell 8: Plots + Summary (with combined overlays) =====================
# Purpose: Visualize training/eval curves and compute final CE/PPL for each fine-tuned model.
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import pandas as pd
from torch.utils.data import DataLoader

def _extract_logs(trainer):
    # Pull out loss, eval_loss, and LR vs. steps from TRL's log history
    logs = getattr(trainer.state, "log_history", [])
    loss_s, loss_v, lr_s, lr_v, eval_s, eval_v = [], [], [], [], [], []
    step_counter = 0
    for row in logs:
        step = int(row.get("step", step_counter))
        step_counter = max(step_counter, step)
        if "loss" in row:
            loss_s.append(step); loss_v.append(float(row["loss"]))
        if "learning_rate" in row:
            lr_s.append(step); lr_v.append(float(row["learning_rate"]))
        if "eval_loss" in row:
            eval_s.append(step); eval_v.append(float(row["eval_loss"]))
    return loss_s, loss_v, lr_s, lr_v, eval_s, eval_v

# Per-model curves
for name, obj in FT_RUNS.items():
    loss_s, loss_v, lr_s, lr_v, eval_s, eval_v = _extract_logs(obj["trainer"])
    display(Markdown(f"### {name}"))
    if loss_s:
        plt.figure(figsize=(7.5, 4))
        plt.plot(loss_s, loss_v, marker='o', linewidth=1)
        plt.title(f"{name} — Training loss")
        plt.xlabel("Step"); plt.ylabel("Loss"); plt.grid(True, alpha=.35)
        plt.tight_layout(); plt.show()
    if lr_s:
        plt.figure(figsize=(7.5, 4))
        plt.plot(lr_s, lr_v, linewidth=1.2)
        plt.title(f"{name} — Learning rate")
        plt.xlabel("Step"); plt.ylabel("LR"); plt.grid(True, alpha=.35)
        plt.tight_layout(); plt.show()
    if eval_s:
        plt.figure(figsize=(7.5, 4))
        plt.plot(eval_s, eval_v, marker='o', linewidth=1)
        plt.title(f"{name} — Eval loss")
        plt.xlabel("Step"); plt.ylabel("Eval loss"); plt.grid(True, alpha=.35)
        plt.tight_layout(); plt.show()

# Overlays: training and eval losses across models
plt.figure(figsize=(8.5, 4.5))
any_series = False
for name, obj in FT_RUNS.items():
    loss_s, loss_v, _, _, _, _ = _extract_logs(obj["trainer"])
    if loss_s:
        plt.plot(loss_s, loss_v, marker='o', linewidth=1, label=name)
        any_series = True
if any_series:
    plt.title("Overlay — Training loss (all models)")
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.grid(True, alpha=.35); plt.legend(loc="best", frameon=True)
    plt.tight_layout(); plt.show()
else:
    display(Markdown("⚠️ No training loss logs to overlay."))

plt.figure(figsize=(8.5, 4.5))
any_series = False
for name, obj in FT_RUNS.items():
    _, _, _, _, eval_s, eval_v = _extract_logs(obj["trainer"])
    if eval_s:
        plt.plot(eval_s, eval_v, marker='o', linewidth=1, label=name)
        any_series = True
if any_series:
    plt.title("Overlay — Eval loss (all models)")
    plt.xlabel("Step"); plt.ylabel("Eval loss")
    plt.grid(True, alpha=.35); plt.legend(loc="best", frameon=True)
    plt.tight_layout(); plt.show()
else:
    display(Markdown("⚠️ No eval loss logs to overlay."))

# Compute final CE/PPL for each fine-tuned model and summarize
summary = {}
for name, obj in FT_RUNS.items():
    tok = obj["tok"]
    tnm = tokenize_and_mask_factory(tok)
    val_tok  = val_sft.map(tnm, remove_columns=["text"])
    test_tok = test_sft.map(tnm, remove_columns=["text"])
    coll = pad_collate(tok)
    val_loader  = DataLoader(val_tok,  batch_size=2, shuffle=False, collate_fn=coll)
    test_loader = DataLoader(test_tok, batch_size=2, shuffle=False, collate_fn=coll)
    val_ce,  val_ppl  = eval_ce_ppl(obj["trainer"].model, val_loader)
    test_ce, test_ppl = eval_ce_ppl(obj["trainer"].model, test_loader)
    summary[name] = dict(val_ce=val_ce, val_ppl=val_ppl,
                         test_ce=test_ce, test_ppl=test_ppl)

df = (pd.DataFrame(summary).T
      .rename_axis("model").reset_index()
      .sort_values(by="test_ce", ascending=True))
display(Markdown("## 📊 **Summary (fine-tuned models)** — sorted by test CE"))
display(df.style.format({
    "val_ce": "{:.4f}", "val_ppl": "{:.2f}",
    "test_ce": "{:.4f}", "test_ppl": "{:.2f}"
}))

# Quick bars (lower = better)
plt.figure(figsize=(9.5, 4.5))
plt.bar(df["model"], df["test_ce"])
plt.title("Test Cross-Entropy — all models (lower is better)")
plt.ylabel("CE"); plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", alpha=0.3); plt.tight_layout(); plt.show()

plt.figure(figsize=(9.5, 4.5))
plt.bar(df["model"], df["test_ppl"])
plt.title("Test Perplexity — all models (lower is better)")
plt.ylabel("PPL"); plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", alpha=0.3); plt.tight_layout(); plt.show()

# ===================== Baseline sweep (no training) — VAL ONLY =====================
# Purpose: Re-run baseline (VAL and optional TEST) cleanly and visualize + pick the best by Test CE if present.
import gc, torch, pandas as pd
from torch.utils.data import DataLoader

def free_cuda():
    try:
        torch.cuda.empty_cache()
        gc.collect()
    except:
        pass

baseline_rows = []

if RUN_BASELINE:
    print("Running baseline (no FT) on:", MODEL_NAMES)
    for name in MODEL_NAMES:
        print(f"\n=== Baseline eval: {name} ===")
        tok, model = load_model_and_tokenizer(name, hf_token=HF_TOKEN)
        if tok is None or model is None:
            print(f"[SKIP] Could not load {name}")
            continue

        # tokenize + collate for VAL
        tnm = tokenize_and_mask_factory(tok)
        val_tok = val_sft.map(tnm, remove_columns=["text"])
        coll   = pad_collate(tok)
        val_loader = DataLoader(val_tok, batch_size=2, shuffle=False, collate_fn=coll)

        # metrics (validation)
        val_ce, val_ppl = eval_ce_ppl(model, val_loader)
        row = {"model": name, "val_ce": val_ce, "val_ppl": val_ppl}

        # optional: test
        if 'EVAL_ON_TEST' in globals() and EVAL_ON_TEST:
            test_tok = test_sft.map(tnm, remove_columns=["text"])
            test_loader = DataLoader(test_tok, batch_size=2, shuffle=False, collate_fn=coll)
            test_ce, test_ppl = eval_ce_ppl(model, test_loader)
            row.update({"test_ce": test_ce, "test_ppl": test_ppl})
            print(f"Validation — CE: {val_ce:.4f} | PPL: {val_ppl:.2f} | n={len(val_tok)}")
            print(f"Test       — CE: {test_ce:.4f} | PPL: {test_ppl:.2f} | n={len(test_tok)}")
        else:
            print(f"Validation — CE: {val_ce:.4f} | PPL: {val_ppl:.2f} | n={len(val_tok)}")

        baseline_rows.append(row)

        # free VRAM between models
        try: del model
        except: pass
        free_cuda()

# Make DataFrame (sorted by validation CE); can contain test_* columns if you enabled EVAL_ON_TEST
baseline_df = pd.DataFrame(baseline_rows)
if not baseline_df.empty:
    sort_key = "val_ce"
    baseline_df = baseline_df.sort_values(sort_key, ascending=True).reset_index(drop=True)
baseline_df

# ===================== Baseline plots + pick winner =====================
# Purpose: Pretty table + bars; select best baseline by Test CE when available.
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

assert 'baseline_df' in globals() and not baseline_df.empty, "No baseline results to plot."

display(Markdown("## 📊 **Baseline (no fine-tune) — Test only**")) # Title reflects test-first if present
display(baseline_df.style.format({
    "val_ce": "{:.4f}", "val_ppl": "{:.2f}",
    **({"test_ce": "{:.4f}", "test_ppl": "{:.2f}"} if "test_ce" in baseline_df.columns else {})
}))

# Bars for Test metrics (if computed)
if "test_ce" in baseline_df.columns:
    plt.figure(figsize=(9, 4.5))
    plt.bar(baseline_df["model"], baseline_df["test_ce"])
    plt.title("Test Cross-Entropy — baseline models (lower is better)")
    plt.ylabel("CE"); plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout(); plt.show()
else:
    display(Markdown("⚠️ Test CE results not available for plotting."))

if "test_ppl" in baseline_df.columns:
    plt.figure(figsize=(9, 4.5))
    plt.bar(baseline_df["model"], baseline_df["test_ppl"])
    plt.title("Test Perplexity — baseline models (lower is better)")
    plt.ylabel("PPL"); plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout(); plt.show()
else:
     display(Markdown("⚠️ Test PPL results not available for plotting."))

# Winner selection by Test CE if present; else warn.
if "test_ce" in baseline_df.columns:
    baseline_df_sorted_test = baseline_df.sort_values("test_ce", ascending=True).reset_index(drop=True)
    best_row = baseline_df_sorted_test.iloc[0]
    display(Markdown(
        f"### ✅ **Selected base model (no FT):** `{best_row['model']}`  \n"
        f"- Test CE: **{best_row['test_ce']:.4f}**, PPL: **{best_row['test_ppl']:.2f}**"
        + (f"\n- Validation CE: **{best_row['val_ce']:.4f}**, PPL: **{best_row['val_ppl']:.2f}**" if "val_ce" in baseline_df.columns else "")
    ))
else:
    display(Markdown("### ⚠️ Cannot select winner based on Test CE as test results are not available."))

# ===================== DeepSeek Baseline on COMMIT dataset =====================
# Purpose: Evaluate DeepSeek 1.3B Instruct on your CommitPackFT (Python) compact splits before any training.
# The cell synthesizes targets from subject/message when explicit 'target' is absent.
import os, math, gc, random, torch, pandas as pd
from datasets import load_from_disk, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

SEED          = 42
MODEL_NAME    = "deepseek-ai/deepseek-coder-1.3b-instruct"
COMMIT_DIR    = "/content/drive/MyDrive/commitpackft_compact_by_lang/python"  # path to your per-lang compact
EVAL_ON_TEST  = True

random.seed(SEED); torch.manual_seed(SEED)
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

assert os.path.exists(COMMIT_DIR), f"Missing commit dataset at: {COMMIT_DIR}"
ds_commit: DatasetDict = load_from_disk(COMMIT_DIR)
print("Commit splits:", {k: len(ds_commit[k]) for k in ds_commit.keys()})

# Synthesize target if needed; filter for minimally well-formed diffs/targets
def _synth_target(ex):
    subj = (ex.get("subject") or "").strip()
    msg  = (ex.get("message") or "").strip()
    if msg and msg.lower() != subj.lower():
        s = (subj + " " + msg).strip() if subj else msg
    else:
        s = subj or msg
    return s if s else None

def has_min_fields(ex):
    before = (ex.get("old_code") or ex.get("old_contents") or "").strip()
    after  = (ex.get("new_code") or ex.get("new_contents") or "").strip()
    tgt    = (ex.get("target") or _synth_target(ex) or "").strip()
    return bool(before) and bool(after) and len(tgt) >= 10

def to_chat_commit(ex):
    before = (ex.get("old_code") or ex.get("old_contents") or "").strip()
    after  = (ex.get("new_code") or ex.get("new_contents") or "").strip()
    target = (ex.get("target") or _synth_target(ex) or "").strip()
    system = ("You are a senior engineer. Explain the functional change between two code revisions "
              "in plain English, 1–3 sentences, no jargon, no code, start with a verb.")
    user = ("Language: code\n"
            "Instruction: Explain clearly and concisely what changed between BEFORE and AFTER.\n"
            "BEFORE:\n```diff\n" + before + "\n```\n\n"
            "AFTER:\n```diff\n" + after + "\n```")
    return {"text":
        "<|im_start|>system\n" + system + "<|im_end|>\n" +
        "<|im_start|>user\n"   + user   + "<|im_end|>\n" +
        "<|im_start|>assistant\n" + target + "<|im_end|>\n"
    }

# Prepare validation/test splits in chat format
commit_train = ds_commit.get("train") or ds_commit[list(ds_commit.keys())[0]]
commit_val   = ds_commit.get("validation") or ds_commit[list(ds_commit.keys())[1]]
commit_test  = ds_commit.get("test") if "test" in ds_commit else None

commit_val   = commit_val.filter(has_min_fields)
commit_sft_v = commit_val.map(to_chat_commit, remove_columns=commit_val.column_names)

commit_sft_t = None
if EVAL_ON_TEST and commit_test is not None:
    commit_test  = commit_test.filter(has_min_fields)
    commit_sft_t = commit_test.map(to_chat_commit, remove_columns=commit_test.column_names)

print("Prepared (commit):", {"val": len(commit_sft_v), "test": (len(commit_sft_t) if commit_sft_t else 0)})

# Loader/tokenizer + masking (assistant-only)
MAX_LEN = 1536
RESPONSE_TAG = "<|im_start|>assistant\n"
ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

def load_model_and_tokenizer(model_name: str, hf_token=None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16
    )
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", quantization_config=bnb,
        trust_remote_code=True, token=hf_token
    )
    model.config.use_cache = False
    return tok, model

def tokenize_and_mask_factory(tok):
    def _f(ex):
        full = ex["text"]
        assert RESPONSE_TAG in full, "assistant tag not found"
        prefix, _ = full.split(RESPONSE_TAG, 1)
        prompt_only = prefix + RESPONSE_TAG
        prompt_ids = tok(prompt_only, truncation=True, max_length=MAX_LEN, add_special_tokens=True)["input_ids"]
        prompt_len = len(prompt_ids)
        enc = tok(full, truncation=True, max_length=MAX_LEN, add_special_tokens=True)
        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
        labels = input_ids.copy()
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return _f

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

@torch.no_grad()
def eval_ce_ppl(model, loader):
    model.eval()
    total_loss, total_items = 0.0, 0
    for batch in loader:
        ids = batch["input_ids"].to(model.device)
        attn = batch["attention_mask"].to(model.device)
        lbs  = batch["labels"].to(model.device)
        logits = model(input_ids=ids, attention_mask=attn).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = lbs[:, 1:].contiguous()
        loss_tok = ce_loss(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1)).view(shift_labels.size(0), -1)
        valid = (shift_labels != -100).float()
        loss_per_ex = (loss_tok * valid).sum(1) / (valid.sum(1) + 1e-9)
        total_loss += float(loss_per_ex.sum())
        total_items += loss_per_ex.numel()
    avg_ce = total_loss / max(total_items, 1)
    ppl = math.exp(avg_ce) if avg_ce < 20 else float("inf")
    return avg_ce, ppl

def free_cuda():
    try:
        torch.cuda.empty_cache(); gc.collect()
    except:
        pass

# Baseline eval on Commit (VAL + optional TEST)
print(f"\n=== Baseline eval (no training): {MODEL_NAME} on COMMIT ===")
tok, model = load_model_and_tokenizer(MODEL_NAME)
tnm = tokenize_and_mask_factory(tok)
val_tok  = commit_sft_v.map(tnm, remove_columns=["text"])
coll     = pad_collate(tok)
val_loader  = DataLoader(val_tok,  batch_size=2, shuffle=False, collate_fn=coll)

val_ce, val_ppl = eval_ce_ppl(model, val_loader)
row = {"model": MODEL_NAME, "val_ce": val_ce, "val_ppl": val_ppl}

if EVAL_ON_TEST and commit_sft_t is not None and len(commit_sft_t) > 0:
    test_tok   = commit_sft_t.map(tnm, remove_columns=["text"])
    test_loader = DataLoader(test_tok, batch_size=2, shuffle=False, collate_fn=coll)
    test_ce, test_ppl = eval_ce_ppl(model, test_loader)
    row.update({"test_ce": test_ce, "test_ppl": test_ppl})
    print(f"Validation — CE: {val_ce:.4f} | PPL: {val_ppl:.2f} | n={len(val_tok)}")
    print(f"Test       — CE: {test_ce:.4f} | PPL: {test_ppl:.2f} | n={len(test_tok)}")
else:
    print(f"Validation — CE: {val_ce:.4f} | PPL: {val_ppl:.2f} | n={len(val_tok)}")

df = pd.DataFrame([row])
from IPython.display import display, Markdown
display(Markdown("## 📊 DeepSeek baseline on Commit dataset"))
display(df.style.format({
    "val_ce": "{:.4f}", "val_ppl": "{:.2f}",
    **({"test_ce": "{:.4f}", "test_ppl": "{:.2f}"} if "test_ce" in df.columns else {})
}))
del model
free_cuda()

# ===================== DeepSeek Baseline on COCKTAIL (50/50) — ROBUST =====================
# Purpose: Validate DeepSeek 1.3B on a mixed CodeSearchNet/Commit 50/50 dataset using the same chat+masking pipeline.
import os, math, gc, random, torch, pandas as pd
from datasets import load_from_disk, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from IPython.display import display, Markdown

SEED          = 42
MODEL_NAME    = "deepseek-ai/deepseek-coder-1.3b-instruct"
COCKTAIL_DIR  = "/content/drive/MyDrive/cocktail_50_50_python"
EVAL_ON_TEST  = True

random.seed(SEED); torch.manual_seed(SEED)
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

assert os.path.exists(COCKTAIL_DIR), f"Missing cocktail dataset at: {COCKTAIL_DIR}"
ds_cocktail: DatasetDict = load_from_disk(COCKTAIL_DIR)
print("Cocktail splits:", {k: len(ds_cocktail[k]) for k in ds_cocktail.keys()})

# Unify raw -> chat across both CodeSearchNet-like and Commit-like rows
SYS = ("You are a senior engineer. Summarize for non-technical managers: "
       "plain English, 1–3 sentences, no jargon, no code, start with a verb.")

def _synth_commit_target(ex):
    subj = (ex.get("subject") or "").strip()
    msg  = (ex.get("message") or "").strip()
    if msg and msg.lower() != subj.lower():
        return (subj + " " + msg).strip() if subj else msg
    return subj or msg

def _good_example_raw(ex):
    if ex.get("code") is not None:
        tgt = (ex.get("docstring") or "").strip()
        return bool(tgt) and len(tgt) >= 10 and tgt[0].isalpha()
    before = (ex.get("old_code") or ex.get("old_contents") or "").strip()
    after  = (ex.get("new_code") or ex.get("new_contents") or "").strip()
    tgt    = (ex.get("target") or _synth_commit_target(ex) or "").strip()
    return bool(before) and bool(after) and len(tgt) >= 10 and tgt[0].isalpha()

def _to_chat(ex):
    if ex.get("code") is not None:  # CodeSearchNet
        code   = (ex.get("code") or "").strip()
        target = (ex.get("docstring") or "").strip()
        user = ("Language: python\n"
                "Instruction: Explain clearly and concisely what this code does.\n"
                f"Code:\n```python\n{code}\n```")
        tgt = target
    else:                           # Commit
        before = (ex.get("old_code") or ex.get("old_contents") or "").strip()
        after  = (ex.get("new_code") or ex.get("new_contents") or "").strip()
        target = (ex.get("target") or _synth_commit_target(ex) or "").strip()
        user = ("Language: python\n"
                "Instruction: Explain clearly and concisely what changed between BEFORE and AFTER.\n"
                f"BEFORE:\n```python\n{before}\n```\n\n"
                f"AFTER:\n```python\n{after}\n```")
        tgt = target
    return {"text":
        "<|im_start|>system\n" + SYS + "<|im_end|>\n" +
        "<|im_start|>user\n"   + user + "<|im_end|>\n" +
        "<|im_start|>assistant\n" + tgt + "<|im_end|>\n"
    }

def ensure_chat_split(ds_split):
    # If already chat-formatted, return as-is; else filter -> map to chat text.
    if ds_split is None:
        return None
    cols = set(ds_split.column_names)
    if "text" in cols:
        if len(ds_split) == 0:
            return ds_split
        sample = ds_split[0].get("text", "")
        if "<|im_start|>assistant\n" not in sample:
            print("⚠️ Split has 'text' but missing assistant tag in first row; proceeding anyway.")
        return ds_split
    ds_f = ds_split.filter(_good_example_raw)
    if len(ds_f) == 0:
        print("⚠️ After filtering, no rows left; returning original split (might already be curated).")
        return ds_split
    return ds_f.map(_to_chat, remove_columns=ds_f.column_names)

val_chat = ensure_chat_split(ds_cocktail.get("validation"))
test_chat = ensure_chat_split(ds_cocktail.get("test") if "test" in ds_cocktail else None)
print("Prepared (cocktail):", {"val": len(val_chat) if val_chat is not None else 0,
                               "test": len(test_chat) if test_chat is not None else 0})
assert val_chat is not None and "text" in val_chat.column_names and len(val_chat) > 0, "Validation split has no 'text' examples."

# Tokenize/mask + CE/PPL
MAX_LEN = 1536
RESPONSE_TAG = "<|im_start|>assistant\n"
ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

def load_model_and_tokenizer(model_name: str, hf_token=None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16
    )
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", quantization_config=bnb,
        trust_remote_code=True, token=hf_token
    )
    model.config.use_cache = False
    return tok, model

def tokenize_and_mask_factory(tok):
    def _f(ex):
        full = ex["text"]
        if RESPONSE_TAG not in full:
            enc = tok(full, truncation=True, max_length=MAX_LEN, add_special_tokens=True)
            L = len(enc["input_ids"])
            return {"input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "labels": [-100]*L}
        prefix, _ = full.split(RESPONSE_TAG, 1)
        prompt_only = prefix + RESPONSE_TAG
        prompt_ids = tok(prompt_only, truncation=True, max_length=MAX_LEN, add_special_tokens=True)["input_ids"]
        prompt_len = len(prompt_ids)
        enc = tok(full, truncation=True, max_length=MAX_LEN, add_special_tokens=True)
        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
        labels = input_ids.copy()
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return _f

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

@torch.no_grad()
def eval_ce_ppl(model, loader):
    model.eval()
    total_loss, total_items = 0.0, 0
    for batch in loader:
        ids  = batch["input_ids"].to(model.device)
        attn = batch["attention_mask"].to(model.device)
        lbs  = batch["labels"].to(model.device)
        logits = model(input_ids=ids, attention_mask=attn).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = lbs[:, 1:].contiguous()
        loss_tok = ce_loss(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1)).view(shift_labels.size(0), -1)
        valid = (shift_labels != -100).float()
        loss_per_ex = (loss_tok * valid).sum(1) / (valid.sum(1) + 1e-9)
        total_loss += float(loss_per_ex.sum()); total_items += loss_per_ex.numel()
    avg_ce = total_loss / max(total_items, 1)
    ppl = math.exp(avg_ce) if avg_ce < 20 else float("inf")
    return avg_ce, ppl

def free_cuda():
    try:
        torch.cuda.empty_cache(); gc.collect()
    except:
        pass

print(f"\n=== Baseline eval (no training): {MODEL_NAME} on COCKTAIL ===")
tok, model = load_model_and_tokenizer(MODEL_NAME)
tnm = tokenize_and_mask_factory(tok)
val_tok   = val_chat.map(tnm, remove_columns=[c for c in val_chat.column_names if c != "text"]).remove_columns(["text"])
coll      = pad_collate(tok)
val_loader  = DataLoader(val_tok,  batch_size=2, shuffle=False, collate_fn=coll)

val_ce, val_ppl = eval_ce_ppl(model, val_loader)
row = {"dataset": "Cocktail 50/50", "model": MODEL_NAME, "val_ce": val_ce, "val_ppl": val_ppl}

if EVAL_ON_TEST and (test_chat is not None) and len(test_chat) > 0:
    test_tok = test_chat.map(tnm, remove_columns=[c for c in test_chat.column_names if c != "text"]).remove_columns(["text"])
    test_loader = DataLoader(test_tok, batch_size=2, shuffle=False, collate_fn=coll)
    test_ce, test_ppl = eval_ce_ppl(model, test_loader)
    row.update({"test_ce": test_ce, "test_ppl": test_ppl})
    print(f"Validation — CE: {val_ce:.4f} | PPL: {val_ppl:.2f} | n={len(val_tok)}")
    print(f"Test       — CE: {test_ce:.4f} | PPL: {test_ppl:.2f} | n={len(test_tok)}")
else:
    print(f"Validation — CE: {val_ce:.4f} | PPL: {val_ppl:.2f} | n={len(val_tok)}")

df = pd.DataFrame([row])
display(Markdown("## 📊 DeepSeek baseline on Cocktail 50/50"))
display(df.style.format({
    "val_ce": "{:.4f}", "val_ppl": "{:.2f}",
    **({"test_ce": "{:.4f}", "test_ppl": "{:.2f}"} if "test_ce" in df.columns else {})
}))

import matplotlib.pyplot as plt
plt.figure(figsize=(6.4, 3.8))
plt.bar(["Validation"], [row["val_ce"]], width=0.5)
plt.title("Validation CE — Cocktail"); plt.ylabel("CE"); plt.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.show()

if "test_ce" in row:
    plt.figure(figsize=(6.4, 3.8))
    plt.bar(["Test"], [row["test_ce"]], width=0.5)
    plt.title("Test CE — Cocktail"); plt.ylabel("CE"); plt.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.show()

# Free VRAM
del model
free_cuda()

# ===================== Tiny comparison plot (user-provided numbers) =====================
# Purpose: Quick visualization of aggregate baseline metrics across three datasets.
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "dataset": ["Original", "CommitPackFT", "Cocktail 50/50"],
    "val_ce": [0.7850, 2.7596, 1.7609],
    "val_ppl": [2.07, 16.72, 6.01]
}
df_comparison = pd.DataFrame(data)
display(df_comparison)

plt.figure(figsize=(8, 5))
plt.bar(df_comparison["dataset"], df_comparison["val_ce"])
plt.title("DeepSeek Baseline Validation CE across Datasets")
plt.xlabel("Dataset"); plt.ylabel("Validation Cross-Entropy")
plt.grid(axis="y", alpha=0.3); plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 5))
plt.bar(df_comparison["dataset"], df_comparison["val_ppl"])
plt.title("DeepSeek Baseline Validation PPL across Datasets")
plt.xlabel("Dataset"); plt.ylabel("Validation Perplexity")
plt.grid(axis="y", alpha=0.3); plt.tight_layout(); plt.show()
