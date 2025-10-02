# ===== Cell 1: Mount + imports + dataset load =====
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, random, math, gc, torch
from datasets import load_from_disk

# Repro
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Load the compact CodeSearchNet (python-only) dataset saved to Drive.
# We assert it exists to fail fast with a clear error if the path is wrong.
py_dir = "/content/drive/MyDrive/codesearchnet_compact_by_lang/python"
assert os.path.exists(py_dir), f"Missing dataset at: {py_dir}"
ds_python = load_from_disk(py_dir)
print("Splits:", list(ds_python.keys()))
print({k: len(ds_python[k]) for k in ds_python.keys()})

# Disable external tracking and nudge CUDA to fragment less.
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ===== Cell 2: Prepare training/eval texts =====
# We keep only examples with a non-trivial docstring (10+ chars and alphabetic start).
def good_doc(ex):
    t = (ex.get("docstring") or "").strip()
    return len(t) >= 10 and t[0].isalpha()

# Filter each split to “good” examples only.
train_raw = ds_python["train"].filter(good_doc)
val_raw   = ds_python["validation"].filter(good_doc)
test_raw  = ds_python["test"].filter(good_doc)
print("Prepared filter sizes:", len(train_raw), len(val_raw), len(test_raw))

# System instruction (stable across all examples).
SYS = ("You are a senior engineer. Summarize code for non-technical managers: "
       "plain English, 1–3 sentences, no jargon, no code, start with a verb.")

# Convert raw example -> chat-style string with system/user/assistant turns.
def to_chat(ex):
    code = ex["code"]
    target = (ex.get("docstring") or "").strip()
    user = (f"Language: python\n"
            f"Instruction: Explain clearly and concisely what this code does.\n"
            f"Code:\n```python\n{code}\n```")
    return {"text":
        "<|im_start|>system\n" + SYS + "<|im_end|>\n" +
        "<|im_start|>user\n" + user + "<|im_end|>\n" +
        "<|im_start|>assistant\n" + target + "<|im_end|>\n"
    }

# Map each split into chat-format and drop original columns.
train_sft = train_raw.map(to_chat, remove_columns=train_raw.column_names)
val_sft   = val_raw.map(to_chat,   remove_columns=val_raw.column_names)
test_sft  = test_raw.map(to_chat,  remove_columns=test_raw.column_names)

print({"train": len(train_sft), "val": len(val_sft), "test": len(test_sft)})

# ===== Cell 3: Tokenize/Collate + CE/PPL eval =====
from torch.utils.data import DataLoader

# Everything after this tag is considered "assistant output" and gets the label,
# while tokens before are masked (-100) so loss is computed only on the answer.
response_template = "<|im_start|>assistant\n"
MAX_LEN_DEFAULT = 1536

# Tokenize prompt+answer, then mask the prompt tokens in labels to -100.
def tokenize_and_mask_factory(tok, max_len=MAX_LEN_DEFAULT):
    def _f(ex):
        full_text = ex["text"]
        prefix, _ = full_text.split(response_template, 1)
        prompt_only = prefix + response_template

        prompt_ids = tok(prompt_only, truncation=True, max_length=max_len, add_special_tokens=True)["input_ids"]
        prompt_len = len(prompt_ids)

        enc = tok(full_text, truncation=True, max_length=max_len, add_special_tokens=True)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        labels = input_ids.copy()
        cut = min(prompt_len, len(labels))
        for i in range(cut):
            labels[i] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return _f

# Pad variable-length batches for input/attention/labels.
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

# Cross-Entropy on assistant-only tokens; returns average CE and PPL.
ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

@torch.no_grad()
def eval_ce_ppl(model, loader):
    model.eval()
    total_loss, total_items = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(model.device)
        attn      = batch["attention_mask"].to(model.device)
        labels    = batch["labels"].to(model.device)

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_tok = ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.size(0), -1)

        valid = (shift_labels != -100).float()
        loss_per_ex = (loss_tok * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-9)

        total_loss += float(loss_per_ex.sum())
        total_items += loss_per_ex.numel()

    avg_ce = total_loss / max(total_items, 1)
    ppl = math.exp(avg_ce) if avg_ce < 20 else float("inf")
    return avg_ce, ppl

# ===== Cell 4: Model loader + LoRA wrapper =====
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

HF_TOKEN = None  # paste if you need gated models; else keep None

# Load tokenizer + 4-bit quantized model for memory efficiency in Colab.
def load_model_and_tokenizer(model_name: str, hf_token=None):
    try:
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
        model.config.use_cache = False
        return tok, model
    except Exception as e:
        raise RuntimeError(f"Could not load {model_name}: {e}")

# Wrap the base model with LoRA adapters (keeps base weights frozen).
def lora_wrap(model, use_gc=False):
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gc)
    if hasattr(model, "gradient_checkpointing_enable") and use_gc:
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    return get_peft_model(model, lcfg)

def free_cuda():
    torch.cuda.empty_cache()
    gc.collect()

# ===== Cell 5: Hyperparameter sweep — num_train_epochs =====
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from IPython.display import display, Markdown
import pandas as pd
import matplotlib.pyplot as plt

MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"
tok, base_model = load_model_and_tokenizer(MODEL, hf_token=HF_TOKEN)

# Pre-tokenize val/test once and reuse across runs.
tnm = tokenize_and_mask_factory(tok, max_len=1536)
val_tok  = val_sft.map(tnm,  remove_columns=["text"])
test_tok = test_sft.map(tnm, remove_columns=["text"])
coll_eval = pad_collate(tok)
val_loader  = DataLoader(val_tok,  batch_size=2, shuffle=False, collate_fn=coll_eval)
test_loader = DataLoader(test_tok, batch_size=2, shuffle=False, collate_fn=coll_eval)

EPOCH_GRID = [1, 2, 3]     # sweep this
LR_FIXED   = 2e-4          # keep lr fixed in this sweep
BATCH_SIZE = 2             # small per-device batch for Colab
GRAD_ACC   = 4
MAX_LEN    = 1536
USE_GC     = False         # set True if you run OOM

results = []      # aggregated final CE/PPL per epoch choice
curves  = {}      # per-run training/eval loss curves

for EPOCHS in EPOCH_GRID:
    print(f"\n=== Training {MODEL} for {EPOCHS} epoch(s) ===")
    # Create a fresh LoRA head for each run (base_model kept in memory).
    model = lora_wrap(base_model, use_gc=USE_GC)

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tok, response_template="<|im_start|>assistant\n"
    )

    cfg = SFTConfig(
        output_dir=f"ft_deepseek13b_ep{EPOCHS}",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR_FIXED,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_seq_length=MAX_LEN,
        packing=False,
        bf16=False,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=10_000,
        report_to="none",
        dataset_text_field="text",
        optim="paged_adamw_32bit",
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tok,
        train_dataset=train_sft, eval_dataset=val_sft,
        args=cfg, data_collator=collator,
    )

    trainer.train()

    # Extract curves from trainer's log history.
    logs = getattr(trainer.state, "log_history", [])
    curves[EPOCHS] = {
        "loss": [(int(r["step"]), float(r["loss"])) for r in logs if "loss" in r],
        "eval_loss": [(int(r["step"]), float(r["eval_loss"])) for r in logs if "eval_loss" in r],
        "lr": [(int(r["step"]), float(r["learning_rate"])) for r in logs if "learning_rate" in r],
    }

    # Compute CE/PPL using our masking metric on val/test.
    val_ce,  val_ppl  = eval_ce_ppl(trainer.model, val_loader)
    test_ce, test_ppl = eval_ce_ppl(trainer.model, test_loader)
    results.append(dict(epochs=EPOCHS, val_ce=val_ce, val_ppl=val_ppl,
                        test_ce=test_ce, test_ppl=test_ppl))
    print(f"[Epochs={EPOCHS}]  Val CE={val_ce:.4f} PPL={val_ppl:.2f} | Test CE={test_ce:.4f} PPL={test_ppl:.2f}")

    # Drop trainer/model objects between runs to free VRAM.
    try:
        del trainer
        del model
    except:
        pass
    free_cuda()

# Summarize sweep results in a table + plots.
df = pd.DataFrame(results).sort_values("test_ce")
display(Markdown("## 📊 **num_train_epochs sweep (DeepSeek 1.3B Instruct)**"))
display(df.style.format({"val_ce": "{:.4f}", "val_ppl": "{:.2f}", "test_ce": "{:.4f}", "test_ppl": "{:.2f}"}))

# Overlay train loss per epoch choice.
plt.figure(figsize=(8,4))
for ep in EPOCH_GRID:
    xs = [s for s,_ in curves[ep]["loss"]]
    ys = [v for _,v in curves[ep]["loss"]]
    if xs:
        plt.plot(xs, ys, marker='o', linewidth=1, label=f"{ep} epoch(s)")
plt.title("Training loss vs steps (by num_train_epochs)")
plt.xlabel("Step"); plt.ylabel("Loss"); plt.grid(True, alpha=.3); plt.legend(); plt.tight_layout(); plt.show()

# Overlay eval loss per epoch choice.
plt.figure(figsize=(8,4))
for ep in EPOCH_GRID:
    xs = [s for s,_ in curves[ep]["eval_loss"]]
    ys = [v for _,v in curves[ep]["eval_loss"]]
    if xs:
        plt.plot(xs, ys, marker='o', linewidth=1, label=f"{ep} epoch(s)")
plt.title("Eval loss vs steps (by num_train_epochs)")
plt.xlabel("Step"); plt.ylabel("Eval loss"); plt.grid(True, alpha=.3); plt.legend(); plt.tight_layout(); plt.show()

# Bar chart: test CE per epoch count.
plt.figure(figsize=(7.5,4))
plt.bar([str(r["epochs"]) for r in results], [r["test_ce"] for r in results])
plt.title("Test Cross-Entropy by num_train_epochs (lower is better)")
plt.xlabel("Epochs"); plt.ylabel("Test CE"); plt.grid(axis="y", alpha=.3); plt.tight_layout(); plt.show()

# ===================== Hyperparameter Cell: learning_rate sweep =====================
import math, gc, os, torch, matplotlib.pyplot as plt, pandas as pd
from IPython.display import display, Markdown
from torch.utils.data import DataLoader
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# Sweep learning rates while holding other knobs.
MODEL_TARGET   = "deepseek-ai/deepseek-coder-1.3b-instruct"
LR_GRID        = [5e-5, 1e-4, 2e-4, 3e-4]
NUM_EPOCHS     = 1
BATCH_SIZE     = 2
GRAD_ACC       = 4
MAX_SEQ_LENGTH = 1536
USE_GC         = True
EVAL_STEPS     = 100
LOGGING_STEPS  = 10

os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Utility to parse trainer logs into arrays for plotting.
def _extract_logs(trainer):
    logs = getattr(trainer.state, "log_history", [])
    loss_s, loss_v, lr_s, lr_v, eval_s, eval_v = [], [], [], [], [], []
    step_counter = 0
    for row in logs:
        step = int(row.get("step", step_counter))
        step_counter = max(step_counter, step)
        if "loss" in row:           loss_s.append(step); loss_v.append(float(row["loss"]))
        if "learning_rate" in row:  lr_s.append(step);   lr_v.append(float(row["learning_rate"]))
        if "eval_loss" in row:      eval_s.append(step); eval_v.append(float(row["eval_loss"]))
    return loss_s, loss_v, lr_s, lr_v, eval_s, eval_v

# Create val/test loaders once for consistent evaluation.
def _prep_loaders(tok, max_len=MAX_SEQ_LENGTH, bs=2):
    tnm = tokenize_and_mask_factory(tok)
    val_tok  = val_sft.map(tnm,  remove_columns=["text"])
    test_tok = test_sft.map(tnm, remove_columns=["text"])
    coll = pad_collate(tok)
    val_loader  = DataLoader(val_tok,  batch_size=bs, shuffle=False, collate_fn=coll)
    test_loader = DataLoader(test_tok, batch_size=bs, shuffle=False, collate_fn=coll)
    return val_loader, test_loader

# Run one training per LR and collect curves + final CE/PPL.
runs = []
for lr in LR_GRID:
    print(f"\n=== learning_rate = {lr:.1e} ===")
    tok, base_model = load_model_and_tokenizer(MODEL_TARGET)
    model = lora_wrap(base_model, use_gc=USE_GC)

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tok, response_template="<|im_start|>assistant\n"
    )

    cfg = SFTConfig(
        output_dir=f"ft_lr_{str(lr).replace('.', 'p')}",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        bf16=False,
        fp16=True,
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=10_000,  # effectively off
        report_to="none",
        dataset_text_field="text",
        optim="paged_adamw_32bit",
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tok,
        train_dataset=train_sft, eval_dataset=val_sft,
        args=cfg, data_collator=collator,
    )

    print("Start training…")
    trainer.train()
    print("Done.")

    runs.append({"lr": lr, "trainer": trainer, "tok": tok})

    try: free_cuda()
    except: pass

# Plot eval-loss vs steps for all LRs.
plt.figure(figsize=(8, 4.5))
for r in runs:
    loss_s, loss_v, _, _, eval_s, eval_v = _extract_logs(r["trainer"])
    if eval_s:
        plt.plot(eval_s, eval_v, marker='o', linewidth=1, label=f"{r['lr']:.1e}")
plt.title("Eval loss vs steps (by learning_rate)")
plt.xlabel("Step"); plt.ylabel("Eval loss")
plt.grid(True, alpha=.35); plt.legend(title="LR"); plt.tight_layout(); plt.show()

# Evaluate final CE/PPL per LR on val & test.
summary = []
for r in runs:
    tok = r["tok"]
    val_loader, test_loader = _prep_loaders(tok, MAX_SEQ_LENGTH, bs=2)
    v_ce, v_ppl   = eval_ce_ppl(r["trainer"].model, val_loader)
    t_ce, t_ppl   = eval_ce_ppl(r["trainer"].model, test_loader)
    summary.append({"learning_rate": r["lr"], "val_ce": v_ce, "val_ppl": v_ppl,
                    "test_ce": t_ce, "test_ppl": t_ppl})

df = (pd.DataFrame(summary)
        .sort_values("test_ce", ascending=True)
        .reset_index(drop=True))
display(Markdown("## 📊 **learning_rate sweep (DeepSeek 1.3B Instruct)**"))
display(df.style.format({"learning_rate": "{:.1e}",
                         "val_ce": "{:.4f}", "val_ppl": "{:.2f}",
                         "test_ce": "{:.4f}", "test_ppl": "{:.2f}"}))

# Bar chart for test CE per LR (lower is better).
plt.figure(figsize=(7.5, 3.8))
plt.bar([f"{lr:.1e}" for lr in df["learning_rate"]], df["test_ce"])
plt.title("Test Cross-Entropy by learning_rate (lower is better)")
plt.xlabel("learning_rate"); plt.ylabel("Test CE")
plt.grid(axis="y", alpha=.3); plt.tight_layout(); plt.show()

# ===================== Hyperparameter Cell: lr_scheduler_type sweep (VALIDATION-ONLY) =====================
import os, gc, torch, math, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown
from torch.utils.data import DataLoader
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# Sweep schedulers while holding LR/warmup constant; evaluate on validation only.
MODEL_TARGET    = "deepseek-ai/deepseek-coder-1.3b-instruct"
LR_FIXED        = float(BEST_LR) if 'BEST_LR' in globals() else 2e-4
SCHEDULERS      = ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup", "constant"]
NUM_EPOCHS      = 1
BATCH_SIZE      = 2
GRAD_ACC        = 4
MAX_SEQ_LENGTH  = 1536
USE_GC          = True
WARMUP_RATIO    = 0.05
EVAL_STEPS      = 100
LOGGING_STEPS   = 10

os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def _extract_logs(trainer):
    logs = getattr(trainer.state, "log_history", [])
    loss_s, loss_v, eval_s, eval_v = [], [], [], []
    step_counter = 0
    for row in logs:
        step = int(row.get("step", step_counter)); step_counter = max(step_counter, step)
        if "loss" in row:      loss_s.append(step); loss_v.append(float(row["loss"]))
        if "eval_loss" in row: eval_s.append(step); eval_v.append(float(row["eval_loss"]))
    return loss_s, loss_v, eval_s, eval_v

def _prep_val_loader(tok, bs=2):
    tnm = tokenize_and_mask_factory(tok)
    val_tok  = val_sft.map(tnm,  remove_columns=["text"])
    coll = pad_collate(tok)
    return DataLoader(val_tok, batch_size=bs, shuffle=False, collate_fn=coll)

runs = []
for sched in SCHEDULERS:
    print(f"\n=== lr_scheduler_type = {sched} | lr = {LR_FIXED:.1e} ===")
    tok, base_model = load_model_and_tokenizer(MODEL_TARGET)
    model = lora_wrap(base_model, use_gc=USE_GC)

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tok, response_template="<|im_start|>assistant\n"
    )

    cfg = SFTConfig(
        output_dir=f"ft_sched_{sched}",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR_FIXED,
        lr_scheduler_type=sched,
        warmup_ratio=WARMUP_RATIO,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        bf16=False,
        fp16=True,
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=10_000,  # effectively off
        report_to="none",
        dataset_text_field="text",
        optim="paged_adamw_32bit",
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tok,
        train_dataset=train_sft, eval_dataset=val_sft,
        args=cfg, data_collator=collator,
    )

    trainer.train()
    runs.append({"sched": sched, "trainer": trainer, "tok": tok})
    try: free_cuda()
    except: pass

# Plot eval-loss curves for each scheduler (validation only).
plt.figure(figsize=(9, 4.6))
for r in runs:
    _, _, es, ev = _extract_logs(r["trainer"])
    if es: plt.plot(es, ev, marker='o', linewidth=1, label=r["sched"])
plt.title("Validation eval loss vs steps (by lr_scheduler_type)")
plt.xlabel("Step"); plt.ylabel("Eval loss")
plt.grid(True, alpha=.35); plt.legend(title="Scheduler"); plt.tight_layout(); plt.show()

# Compute validation CE/PPL for each scheduler and pick the best one by CE.
summary = []
for r in runs:
    tok = r["tok"]
    val_loader = _prep_val_loader(tok, bs=2)
    v_ce, v_ppl = eval_ce_ppl(r["trainer"].model, val_loader)
    summary.append({"lr_scheduler_type": r["sched"], "val_ce": v_ce, "val_ppl": v_ppl})

df = pd.DataFrame(summary).sort_values("val_ce", ascending=True).reset_index(drop_number=True)
BEST_SCHED = df.iloc[0]["lr_scheduler_type"]

display(Markdown(f"## 📊 **lr_scheduler_type sweep (Validation-only)**  \nUsing fixed LR: **{LR_FIXED:.1e}**, warmup_ratio: **{WARMUP_RATIO}**"))
display(df.style.format({"val_ce": "{:.4f}", "val_ppl": "{:.2f}"}))
print(f"\n>>> Selected BEST_SCHED (lowest validation CE): {BEST_SCHED}")

# ===================== Hyperparameter Cell: batch-size sweep (VALIDATION-ONLY) =====================
import os, gc, math, torch, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown
from torch.utils.data import DataLoader
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# Keep LR/scheduler fixed; sweep per-device batch size.
MODEL_TARGET    = "deepseek-ai/deepseek-coder-1.3b-instruct"
LR_FIXED        = float(BEST_LR) if 'BEST_LR' in globals() else 2e-4
SCHEDULER       = str(BEST_SCHED) if 'BEST_SCHED' in globals() else "constant_with_warmup"
WARMUP_RATIO    = 0.05
NUM_EPOCHS      = 1
BATCH_GRID      = [1, 2, 4]
GRAD_ACC_FIXED  = 4
MAX_SEQ_LENGTH  = 1536
USE_GC          = True
EVAL_STEPS      = 100
LOGGING_STEPS   = 10

os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def free_cuda():
    try:
        torch.cuda.empty_cache()
        gc.collect()
    except:
        pass

def _extract_eval_curve(trainer):
    logs = getattr(trainer.state, "log_history", [])
    es, ev = [], []
    step_counter = 0
    for row in logs:
        step = int(row.get("step", step_counter)); step_counter = max(step_counter, step)
        if "eval_loss" in row:
            es.append(step); ev.append(float(row["eval_loss"]))
    return es, ev

def _prep_val_loader(tok, bs=2):
    tnm = tokenize_and_mask_factory(tok)
    val_tok  = val_sft.map(tnm, remove_columns=["text"])
    coll = pad_collate(tok)
    return DataLoader(val_tok, batch_size=bs, shuffle=False, collate_fn=coll)

# Metric function (same as earlier)
ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
@torch.no_grad()
def eval_ce_ppl(model, loader):
    model.eval()
    total_loss, total_items = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(model.device)
        attn      = batch["attention_mask"].to(model.device)
        labels    = batch["labels"].to(model.device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_tok = ce_loss(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1)).view(shift_labels.size(0), -1)
        valid = (shift_labels != -100).float()
        loss_per_ex = (loss_tok * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-9)
        total_loss += float(loss_per_ex.sum())
        total_items += loss_per_ex.numel()
    avg_ce = total_loss / max(total_items, 1)
    ppl = math.exp(avg_ce) if avg_ce < 20 else float("inf")
    return avg_ce, ppl

runs = []
for bsz in BATCH_GRID:
    print(f"\n=== per_device_train_batch_size = {bsz} | lr = {LR_FIXED:.1e} | sched = {SCHEDULER} ===")
    try:
        tok, base_model = load_model_and_tokenizer(MODEL_TARGET)
        model = lora_wrap(base_model, use_gc=USE_GC)

        collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tok, response_template="<|im_start|>assistant\n"
        )

        cfg = SFTConfig(
            output_dir=f"ft_bsz_{bsz}",
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=bsz,
            gradient_accumulation_steps=GRAD_ACC_FIXED,  # fixed for fairness
            learning_rate=LR_FIXED,
            lr_scheduler_type=SCHEDULER,
            warmup_ratio=WARMUP_RATIO,
            max_seq_length=MAX_SEQ_LENGTH,
            packing=False,
            bf16=False,
            fp16=True,
            logging_steps=LOGGING_STEPS,
            evaluation_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_steps=10_000,  # effectively off
            report_to="none",
            dataset_text_field="text",
            optim="paged_adamw_32bit",
        )

        trainer = SFTTrainer(
            model=model, tokenizer=tok,
            train_dataset=train_sft, eval_dataset=val_sft,
            args=cfg, data_collator=collator,
        )

        trainer.train()
        runs.append({"bsz": bsz, "status": "ok", "trainer": trainer, "tok": tok})
    except RuntimeError as e:
        print(f"⚠️ Skipping bsz={bsz} due to error: {e}")
        runs.append({"bsz": bsz, "status": "failed", "trainer": None, "tok": None})
    finally:
        free_cuda()

# Plot eval-loss vs steps for successful runs.
ok_runs = [r for r in runs if r["status"] == "ok" and r["trainer"] is not None]
plt.figure(figsize=(11, 5.6))
for r in ok_runs:
    es, ev = _extract_eval_curve(r["trainer"])
    if es:
        plt.plot(es, ev, linewidth=2.2, label=f"bsz={r['bsz']}")
plt.title("Validation eval loss vs steps (by per-device batch size)")
plt.xlabel("Step"); plt.ylabel("Eval loss")
plt.grid(True, alpha=.35)
if ok_runs:
    plt.legend(title="Batch size", ncols=min(len(ok_runs), 4), loc="upper right", frameon=True)
plt.tight_layout(); plt.show()

# Compute validation CE/PPL for each successful batch size and show a bar chart.
summary = []
for r in ok_runs:
    tok = r["tok"]
    val_loader = _prep_val_loader(tok, bs=2)  # fixed eval batch for fairness
    v_ce, v_ppl = eval_ce_ppl(r["trainer"].model, val_loader)
    summary.append({"batch_size": r["bsz"], "val_ce": v_ce, "val_ppl": v_ppl})

if summary:
    df = pd.DataFrame(summary).sort_values("val_ce", ascending=True).reset_index(drop=True)
    BEST_BSZ = int(df.iloc[0]["batch_size"])

    display(Markdown(
        f"## 📊 **Batch-size sweep (Validation-only)**  \n"
        f"Using LR: **{LR_FIXED:.1e}**, scheduler: **{SCHEDULER}**, warmup_ratio: **{WARMUP_RATIO}**, "
        f"grad_acc (fixed): **{GRAD_ACC_FIXED}**"
    ))
    display(df.style.format({"val_ce": "{:.4f}", "val_ppl": "{:.2f}"}))
    print(f"\n>>> Selected BEST_BSZ (lowest validation CE): {BEST_BSZ}")

    plt.figure(figsize=(7.8, 4.0))
    plt.bar([str(int(x)) for x in df["batch_size"]], df["val_ce"])
    plt.title("Validation Cross-Entropy by per-device batch size (lower is better)")
    plt.xlabel("per_device_train_batch_size"); plt.ylabel("Validation CE")
    plt.grid(axis="y", alpha=.3); plt.tight_layout(); plt.show()
else:
    print("No successful runs to summarize (all batch sizes failed).")

# ===================== Hyperparameter Cell: optimizer sweep incl. SGD (VALIDATION-ONLY) =====================
import os, gc, math, torch, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown
from torch.utils.data import DataLoader
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# Compare several optimizers under the same LR/scheduler/epoch settings.
MODEL_TARGET    = "deepseek-ai/deepseek-coder-1.3b-instruct"
LR_FIXED        = 2e-4
SCHEDULER       = "constant_with_warmup"
WARMUP_RATIO    = 0.05
NUM_EPOCHS      = 1
BATCH_SIZE      = 2
GRAD_ACC        = 4
MAX_SEQ_LENGTH  = 1536
USE_GC          = True
EVAL_STEPS      = 100
LOGGING_STEPS   = 10

OPTIMIZERS = [
    ("paged_adamw_32bit", None),
    ("adamw_torch",       None),
    ("adafactor",         None),
    ("sgd",               "momentum=0.9,nesterov=True"),  # intentionally included for contrast
]

os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def free_cuda():
    try:
        torch.cuda.empty_cache()
        gc.collect()
    except:
        pass

def _extract_eval(trainer):
    logs = getattr(trainer.state, "log_history", [])
    es, ev = [], []
    step_counter = 0
    for row in logs:
        step = int(row.get("step", step_counter)); step_counter = max(step_counter, step)
        if "eval_loss" in row:
            es.append(step); ev.append(float(row["eval_loss"]))
    return es, ev

def _prep_val_loader(tok, bs=2):
    tnm = tokenize_and_mask_factory(tok)
    val_tok  = val_sft.map(tnm, remove_columns=["text"])
    coll = pad_collate(tok)
    return DataLoader(val_tok, batch_size=bs, shuffle=False, collate_fn=coll)

ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
@torch.no_grad()
def eval_ce_ppl(model, loader):
    model.eval()
    total_loss, total_items = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(model.device)
        attn      = batch["attention_mask"].to(model.device)
        labels    = batch["labels"].to(model.device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_tok = ce_loss(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1)).view(shift_labels.size(0), -1)
        valid = (shift_labels != -100).float()
        loss_per_ex = (loss_tok * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-9)
        total_loss += float(loss_per_ex.sum())
        total_items += loss_per_ex.numel()
    avg_ce = total_loss / max(total_items, 1)
    ppl = math.exp(avg_ce) if avg_ce < 20 else float("inf")
    return avg_ce, ppl

runs = []
for opt_name, opt_args in OPTIMIZERS:
    print(f"\n=== optimizer = {opt_name} | lr = {LR_FIXED:.1e} | sched = {SCHEDULER} ===")
    tok, base_model = load_model_and_tokenizer(MODEL_TARGET)
    model = lora_wrap(base_model, use_gc=USE_GC)

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tok, response_template="<|im_start|>assistant\n"
    )

    cfg = SFTConfig(
        output_dir=f"ft_opt_{opt_name}",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR_FIXED,
        lr_scheduler_type=SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        bf16=False,
        fp16=True,
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=10_000,  # effectively off
        report_to="none",
        dataset_text_field="text",
        optim=opt_name,
        optim_args=opt_args,   # e.g., pass SGD momentum
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tok,
        train_dataset=train_sft, eval_dataset=val_sft,
        args=cfg, data_collator=collator,
    )

    trainer.train()
    runs.append({"opt": opt_name, "trainer": trainer, "tok": tok})
    free_cuda()

# Show eval-loss curves by optimizer.
plt.figure(figsize=(11, 5.2), dpi=140)
for r in runs:
    es, ev = _extract_eval(r["trainer"])
    if es:
        label = r["opt"].replace("_", " ")
        plt.plot(es, ev, linewidth=2.0, alpha=0.95, label=label)
plt.title("Validation eval loss vs steps (by optimizer)")
plt.xlabel("Step"); plt.ylabel("Eval loss")
plt.grid(True, alpha=.3); plt.legend(title="Optimizer")
plt.tight_layout(); plt.show()

# Compute validation CE/PPL for each optimizer and select best by CE.
summary = []
for r in runs:
    tok = r["tok"]
    val_loader = _prep_val_loader(tok, bs=2)  # fixed eval bsz for fairness
    v_ce, v_ppl = eval_ce_ppl(r["trainer"].model, val_loader)
    summary.append({"optimizer": r["opt"], "val_ce": v_ce, "val_ppl": v_ppl})

df = pd.DataFrame(summary).sort_values("val_ce", ascending=True).reset_index(drop=True)
BEST_OPT = df.iloc[0]["optimizer"]

display(Markdown(
    f"## 📊 **Optimizer sweep (Validation-only)**  \n"
    f"Using LR: **{LR_FIXED:.1e}**, scheduler: **{SCHEDULER}**, warmup_ratio: **{WARMUP_RATIO}**, "
    f"per-device batch: **{BATCH_SIZE}**, grad_acc: **{GRAD_ACC}**"
))
display(df.style.format({"val_ce": "{:.4f}", "val_ppl": "{:.2f}"}))
print(f"\n>>> Selected BEST_OPT (lowest validation CE): {BEST_OPT}")

# Bar chart of validation CE by optimizer (sorted).
plt.figure(figsize=(8.8, 4.2), dpi=140)
order = df.sort_values("val_ce")["optimizer"]
vals  = df.set_index("optimizer").loc[order]["val_ce"].values
labels = [o.replace("_"," ") for o in order]
bars = plt.bar(labels, vals)
for b, v in zip(bars, vals):
    plt.text(b.get_x()+b.get_width()/2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
plt.title("Validation Cross-Entropy by optimizer (lower is better)")
plt.xlabel("Optimizer"); plt.ylabel("Validation CE")
plt.grid(axis="y", alpha=.3); plt.tight_layout(); plt.show()



#Test Results After Training 
# ===================== Train ON CodeSearchNet TRAIN; Evaluate on TEST (single optimizer) =====================
import os, gc, math, torch, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown
from torch.utils.data import DataLoader
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# Fixed knobs chosen from earlier sweeps (or your preferred picks).
MODEL_TARGET    = "deepseek-ai/deepseek-coder-1.3b-instruct"
LR_FIXED        = 2e-4
SCHEDULER       = "constant_with_warmup"
WARMUP_RATIO    = 0.05
NUM_EPOCHS      = 1
BATCH_SIZE      = 4                 # chosen
GRAD_ACC        = 4                 # chosen
MAX_SEQ_LENGTH  = 1536
USE_GC          = True
EVAL_STEPS      = 100
LOGGING_STEPS   = 10
OPTIMIZER_NAME  = "adafactor"       # single optimizer

os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def free_cuda():
    try:
        torch.cuda.empty_cache()
        gc.collect()
    except:
        pass

def _extract_eval(trainer):
    logs = getattr(trainer.state, "log_history", [])
    es, ev = [], []
    step_counter = 0
    for row in logs:
        step = int(row.get("step", step_counter)); step_counter = max(step_counter, step)
        if "eval_loss" in row:
            es.append(step); ev.append(float(row["eval_loss"]))
    return es, ev

# Build a test loader using our assistant-only masking.
def _prep_test_loader(tok, bs=2):
    tnm = tokenize_and_mask_factory(tok)
    test_tok = test_sft.map(tnm, remove_columns=["text"])
    coll = pad_collate(tok)
    return DataLoader(test_tok, batch_size=bs, shuffle=False, collate_fn=coll)

# CE/PPL on masked labels (assistant-only).
ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
@torch.no_grad()
def eval_ce_ppl(model, loader):
    model.eval()
    total_loss, total_items = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(model.device)
        attn      = batch["attention_mask"].to(model.device)
        labels    = batch["labels"].to(model.device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_tok = ce_loss(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1)).view(shift_labels.size(0), -1)
        valid = (shift_labels != -100).float()
        loss_per_ex = (loss_tok * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-9)
        total_loss += float(loss_per_ex.sum())
        total_items += loss_per_ex.numel()
    avg_ce = total_loss / max(total_items, 1)
    ppl = math.exp(avg_ce) if avg_ce < 20 else float("inf")
    return avg_ce, ppl

# Train on TRAIN; evaluate during training on TEST; report final TEST CE/PPL.
print(f"\n=== Train on CodeSearchNet TRAIN | Eval on TEST | opt={OPTIMIZER_NAME} | lr={LR_FIXED:.1e} | sched={SCHEDULER} | bsz={BATCH_SIZE} ===")
tok, base_model = load_model_and_tokenizer(MODEL_TARGET)
model = lora_wrap(base_model, use_gc=USE_GC)

collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tok, response_template="<|im_start|>assistant\n"
)

cfg = SFTConfig(
    output_dir=f"ft_codesn_train_evaltest_{OPTIMIZER_NAME}",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR_FIXED,
    lr_scheduler_type=SCHEDULER,
    warmup_ratio=WARMUP_RATIO,
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    bf16=False,
    fp16=True,
    logging_steps=LOGGING_STEPS,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,               # evaluate on TEST every EVAL_STEPS
    save_steps=10_000,                   # effectively off
    report_to="none",
    dataset_text_field="text",
    optim=OPTIMIZER_NAME,
)

# IMPORTANT: we intentionally evaluate on TEST here to produce a test-loss curve.
trainer = SFTTrainer(
    model=model, tokenizer=tok,
    train_dataset=train_sft, eval_dataset=test_sft,   # eval on TEST
    args=cfg, data_collator=collator,
)

trainer.train()

# Plot eval-loss on TEST across training steps.
plt.figure(figsize=(11, 5.2), dpi=140)
es, ev = _extract_eval(trainer)
if es:
    plt.plot(es, ev, linewidth=2.0, alpha=0.95, label="Adafactor")
plt.title("Eval loss vs steps (evaluated on TEST)")
plt.xlabel("Step"); plt.ylabel("Eval loss")
plt.grid(True, alpha=.3); plt.legend(title="Optimizer")
plt.tight_layout(); plt.show()

# Final masked CE/PPL on TEST at the end of training.
test_loader = _prep_test_loader(tok, bs=2)
t_ce, t_ppl = eval_ce_ppl(trainer.model, test_loader)
display(Markdown(
    f"## 📊 *CodeSearchNet TEST metrics (trained on TRAIN)*  \n"
    f"- Optimizer: *{OPTIMIZER_NAME}*  \n"
    f"- LR: *{LR_FIXED:.1e}, Scheduler: *{SCHEDULER}*, Epochs: *{NUM_EPOCHS}**, "
    f"per-device batch: *{BATCH_SIZE}, grad_acc: *{GRAD_ACC}**  \n"
    f"- *Test CE:* ⁠ {t_ce:.4f} ⁠  ·  *Test PPL:* ⁠ {t_ppl:.2f} ⁠"
))
