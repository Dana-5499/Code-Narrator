# ===================== Cocktail FT: install + mount + train + report =====================
# --- Clean install (pinned) ---
# Reinstall key libs at specific versions to avoid binary/API mismatches (e.g., NumPy 2.x)
!pip -q uninstall -y numpy transformers tokenizers datasets accelerate trl peft bitsandbytes wandb >/dev/null 2>&1
!pip -q install --no-cache-dir \
  "numpy==1.26.4" \
  "datasets==2.20.0" \
  "transformers==4.44.2" "tokenizers==0.19.1" \
  "accelerate==0.34.2" "trl==0.9.6" "peft==0.12.0" "bitsandbytes==0.47.0"

# Optional cleanup to avoid pulling NumPy>=2.0 via other deps (opencv sometimes drags it in)
!pip -q uninstall -y opencv-python opencv-python-headless opencv-contrib-python thinc >/dev/null 2>&1

# --- Mount Drive ---
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

# --- Imports & config ---
import os, gc, math, random, torch, matplotlib.pyplot as plt, pandas as pd
from IPython.display import display, Markdown
from datasets import load_from_disk, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# Repro + base model + dataset path
SEED          = 42
MODEL_TARGET  = "deepseek-ai/deepseek-coder-1.3b-instruct"
COCKTAIL_DIR  = "/content/drive/MyDrive/cocktail_50_50_python"  # adjust if different

# Hyperparams chosen from prior sweeps
LR_FIXED        = 2e-4
SCHEDULER       = "constant_with_warmup"
WARMUP_RATIO    = 0.05
NUM_EPOCHS      = 1
BATCH_SIZE      = 4
GRAD_ACC        = 4
MAX_SEQ_LENGTH  = 1536
USE_GC          = True
OPTIMIZER_NAME  = "paged_adamw_32bit"   # memory-efficient AdamW

random.seed(SEED); torch.manual_seed(SEED)
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Load cocktail dataset (CodeSearchNet+Commit 50/50) ---
assert os.path.exists(COCKTAIL_DIR), f"Missing cocktail dataset at: {COCKTAIL_DIR}"
ds_cocktail: DatasetDict = load_from_disk(COCKTAIL_DIR)
print("Cocktail splits:", {k: len(ds_cocktail[k]) for k in ds_cocktail.keys()})

# --- Instruction format for unified chat examples (code + commit diffs) ---
SYS = ("You are a senior engineer. Summarize changes or code for non-technical managers: "
       "plain English, 1–3 sentences, no jargon, no code, start with a verb.")
ASSIST = "<|im_start|>assistant\n"

# Helpers to synthesize a reasonable target if only subject/message exists
def _synth_commit_target(ex):
    subj = (ex.get("subject") or "").strip()
    msg  = (ex.get("message") or "").strip()
    if msg and msg.lower() != subj.lower():
        return (subj + " " + msg).strip() if subj else msg
    return subj or msg

def fallback_target(ex):
    # Prefer explicit answer fields; otherwise fall back to synthesized commit text
    t = (ex.get("docstring") or ex.get("target") or "").strip()
    if t:
        return t
    return _synth_commit_target(ex) or ""

def good_example_relaxed(ex, min_len=8):
    # Keep short-but-not-empty targets to avoid over-filtering cocktail
    return len(fallback_target(ex)) >= min_len

# Convert raw to chat format for both kinds: code-only and commit diff
def to_chat_cocktail(ex):
    target = fallback_target(ex).strip()
    if ex.get("code") is not None:  # CodeSearchNet-style record
        code = (ex.get("code") or "").strip()
        user = ("Language: python\n"
                "Instruction: Explain clearly what this code does.\n"
                f"Code:\n```python\n{code}\n```")
    else:  # Commit-style record
        before = (ex.get("old_code") or ex.get("old_contents") or "").strip()
        after  = (ex.get("new_code") or ex.get("new_contents") or "").strip()
        user = ("Language: python\n"
                "Instruction: Explain clearly what changed between BEFORE and AFTER.\n"
                f"BEFORE:\n```python\n{before}\n```\n"
                f"AFTER:\n```python\n{after}\n```")
    return {
        "text":
            "<|im_start|>system\n" + SYS + "<|im_end|>\n" +
            "<|im_start|>user\n"   + user + "<|im_end|>\n" +
            ASSIST + target + "<|im_end|>\n"
    }

# Ensure splits are in chat format (or pass-through if already prepared)
def ensure_chat_split(ds_split):
    if ds_split is None:
        return None
    if "text" in ds_split.column_names and len(ds_split) > 0:
        return ds_split
    ds_f = ds_split.filter(good_example_relaxed)
    if len(ds_f) == 0:
        print("⚠️ After filtering, no rows left; using raw split as-is.")
        ds_f = ds_split
    return ds_f.map(to_chat_cocktail, remove_columns=ds_f.column_names)

train_sft = ensure_chat_split(ds_cocktail.get("train"))
val_sft   = ensure_chat_split(ds_cocktail.get("validation"))
test_sft  = ensure_chat_split(ds_cocktail.get("test") if "test" in ds_cocktail else None)
print("Prepared chat sizes:", {"train": len(train_sft) if train_sft else 0,
                               "val": len(val_sft) if val_sft else 0,
                               "test": len(test_sft) if test_sft else 0})
assert train_sft is not None and len(train_sft) > 0, "No training rows."

# --- Tokenize/Mask helpers (assistant-only loss) for final CE/PPL eval ---
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"
ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

def tokenize_and_mask_factory(tok, max_len=MAX_SEQ_LENGTH):
    # Mask prompt tokens with -100 so CE is computed only on assistant span
    def _f(ex):
        full_text = ex["text"]
        if RESPONSE_TEMPLATE not in full_text:
            enc = tok(full_text, truncation=True, max_length=max_len, add_special_tokens=True)
            L = len(enc["input_ids"])
            return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": [-100]*L}
        prefix, _ = full_text.split(RESPONSE_TEMPLATE, 1)
        prompt_only = prefix + RESPONSE_TEMPLATE
        prompt_ids = tok(prompt_only, truncation=True, max_length=max_len, add_special_tokens=True)["input_ids"]
        prompt_len = len(prompt_ids)
        enc = tok(full_text, truncation=True, max_length=max_len, add_special_tokens=True)
        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
        labels = input_ids.copy()
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return _f

def pad_collate(tok):
    # Left-pad labels with -100, attention with 0, tokens with pad_id
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
    # Compute token-level CE on assistant tokens and report mean CE & PPL
    model.eval()
    tot, n = 0.0, 0
    for b in loader:
        ids, ams, lbs = b["input_ids"].to(model.device), b["attention_mask"].to(model.device), b["labels"].to(model.device)
        logits = model(input_ids=ids, attention_mask=ams).logits
        sl, tl = logits[:, :-1, :], lbs[:, 1:]
        loss_tok = ce_loss(sl.reshape(-1, sl.size(-1)), tl.reshape(-1)).view(tl.size(0), -1)
        valid = (tl != -100).float()
        loss_per_ex = (loss_tok * valid).sum(1) / (valid.sum(1) + 1e-9)
        tot += float(loss_per_ex.sum()); n += loss_per_ex.numel()
    ce = tot / max(n, 1)
    ppl = math.exp(ce) if ce < 20 else float("inf")
    return ce, ppl

def free_cuda():
    try:
        torch.cuda.empty_cache(); gc.collect()
    except:
        pass

# --- Model loader + LoRA (4-bit QLoRA style setup) ---
def load_model_and_tokenizer(model_name: str, hf_token=None):
    # 4-bit NF4 quant to fit 1.3B comfortably on Colab T4/V100
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16
    )
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", quantization_config=bnb,
        trust_remote_code=True, token=hf_token
    )
    model.config.use_cache = False
    return tok, model

def lora_wrap(model, use_gc=False):
    # Prepare k-bit model for PEFT training + attach LoRA adapters to proj modules
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

# ===================== TRAIN on COCKTAIL (TRAIN) / no eval during training =====================
print(f"\n=== Train on Cocktail TRAIN | lr={LR_FIXED:.1e} | sched={SCHEDULER} | bsz={BATCH_SIZE} | optim={OPTIMIZER_NAME} ===")
tok, base_model = load_model_and_tokenizer(MODEL_TARGET)
model = lora_wrap(base_model, use_gc=USE_GC)

# Collator that masks everything before the assistant tag
collator = DataCollatorForCompletionOnlyLM(tokenizer=tok, response_template=ASSIST)

cfg = SFTConfig(
    output_dir=f"ft_cocktail_{OPTIMIZER_NAME}",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR_FIXED,
    lr_scheduler_type=SCHEDULER,
    warmup_ratio=WARMUP_RATIO,
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    bf16=False, fp16=True,
    logging_steps=10,
    evaluation_strategy="no",       # Eval after training to save time/VRAM
    save_steps=10_000,
    report_to="none",
    dataset_text_field="text",
    optim=OPTIMIZER_NAME,
)

trainer = SFTTrainer(
    model=model, tokenizer=tok,
    train_dataset=train_sft, eval_dataset=None,
    args=cfg, data_collator=collator,
)

trainer.train()

# --- Plot: TRAINING loss vs steps (curve for progress) ---
def _extract_train(trainer):
    logs = getattr(trainer.state, "log_history", [])
    xs, ys = [], []
    last_step = 0
    for row in logs:
        if "loss" in row:
            step = int(row.get("step", last_step))
            last_step = max(last_step, step)
            xs.append(step); ys.append(float(row["loss"]))
    return xs, ys

xs, ys = _extract_train(trainer)
plt.figure(figsize=(10.5, 5.2), dpi=140)
if xs:
    plt.plot(xs, ys, linewidth=2.0, alpha=0.95)
plt.title("Training loss vs steps (Cocktail 50/50) — DeepSeek 1.3B LoRA")
plt.xlabel("Step"); plt.ylabel("Training loss")
plt.grid(True, alpha=.3); plt.tight_layout(); plt.show()

# --- Final TEST CE/PPL on cocktail test split (if present) ---
if test_sft is not None and len(test_sft) > 0:
    tnm = tokenize_and_mask_factory(tok)
    test_tok = test_sft.map(tnm, remove_columns=["text"])
    test_loader = DataLoader(test_tok, batch_size=2, shuffle=False, collate_fn=pad_collate(tok))
    t_ce, t_ppl = eval_ce_ppl(trainer.model, test_loader)
    display(Markdown(
        f"## 📊 *Cocktail TEST metrics (trained on TRAIN)*  \n"
        f"- Optimizer: *{OPTIMIZER_NAME}*, LR: *{LR_FIXED:.1e}*, Scheduler: *{SCHEDULER}*, "
        f"Epochs: *{NUM_EPOCHS}*, per-device batch: *{BATCH_SIZE}*, grad_acc: *{GRAD_ACC}*  \n"
        f"- **Test CE:** {t_ce:.4f} · **Test PPL:** {t_ppl:.2f}"
    ))
else:
    display(Markdown("⚠️ No TEST split available to compute final metrics."))

# ===================== Evaluate cocktail-tuned model on CodeSearchNet TEST & CommitPackFT TEST (auto-find ckpt) =====================
import os, gc, math, random, glob, torch, pandas as pd
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from IPython.display import display, Markdown

# Potential output dirs to probe (both /content and Drive)
ROOTS = [
    "/content/ft_cocktail_paged_adamw_32bit",
    "/content/ft_cocktail_adafactor",
    "/content/ft_cocktail",
    "/content/drive/MyDrive/ft_cocktail_paged_adamw_32bit",
    "/content/drive/MyDrive/ft_cocktail_adafactor",
    "/content/drive/MyDrive/ft_cocktail",
]

# Paths for per-task test sets (Python-only compacts)
CODESN_PATH = "/content/drive/MyDrive/codesearchnet_compact_by_lang/python"
COMMIT_PATH = "/content/drive/MyDrive/commitpackft_compact_by_lang/python"
assert os.path.exists(CODESN_PATH), f"Missing: {CODESN_PATH}"
assert os.path.exists(COMMIT_PATH), f"Missing: {COMMIT_PATH}"

SEED = 42
random.seed(SEED); torch.manual_seed(SEED)
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Find a valid checkpoint (prefers full model folder with config.json; otherwise PEFT adapter)
def find_ckpt():
    # 1) exact roots containing a full model
    for r in ROOTS:
        if os.path.isdir(r) and os.path.isfile(os.path.join(r, "config.json")):
            return r, "full"
    # 2) latest checkpoint-* subdir containing full model
    for r in ROOTS:
        if not os.path.isdir(r): continue
        subs = sorted(glob.glob(os.path.join(r, "checkpoint-*")), key=os.path.getmtime, reverse=True)
        for s in subs:
            if os.path.isfile(os.path.join(s, "config.json")):
                return s, "full"
    # 3) latest checkpoint-* with PEFT adapter artifacts
    for r in ROOTS:
        if not os.path.isdir(r): continue
        subs = sorted(glob.glob(os.path.join(r, "checkpoint-*")), key=os.path.getmtime, reverse=True)
        for s in subs:
            if os.path.isfile(os.path.join(s, "adapter_config.json")) or os.path.isfile(os.path.join(s, "adapter_model.safetensors")):
                return s, "peft"
    # 4) root itself has PEFT adapter
    for r in ROOTS:
        if os.path.isdir(r) and (os.path.isfile(os.path.join(r, "adapter_config.json")) or os.path.isfile(os.path.join(r, "adapter_model.safetensors"))):
            return r, "peft"
    raise FileNotFoundError("No usable checkpoint found. Make sure you pass the path that contains either config.json (full) or adapter_config.json (PEFT).")

CKPT_DIR, CKPT_KIND = find_ckpt()
print(f"Using checkpoint: {CKPT_DIR}  [{CKPT_KIND}]")

# Loader that supports either a full saved model or a PEFT adapter on top of the base
BASE_MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"
def load_ckpt(ckpt_dir, kind):
    if kind == "full":
        tok = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True, trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token or tok.unk_token
        tok.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map="auto", trust_remote_code=True)
        model.config.use_cache = False
        return tok, model
    # PEFT route (base in 4-bit + adapter weights)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token or tok.unk_token
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", quantization_config=bnb, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, ckpt_dir)
    model.config.use_cache = False
    return tok, model

tok, model = load_ckpt(CKPT_DIR, CKPT_KIND)

# --- Build CodeSearchNet/Commit chat-formatted TEST sets (Python) ---
codesn = load_from_disk(CODESN_PATH)
commit = load_from_disk(COMMIT_PATH)
test_codesn_raw = codesn["test"]
test_commit_raw = commit["test"]

SYS = ("You are a senior engineer. Summarize code or code changes for non-technical managers: "
       "plain English, 1–3 sentences, no jargon, no code, start with a verb.")
ASSIST = "<|im_start|>assistant\n"

def _clean(s): return (s or "").strip()
def _synth_commit_target(ex):
    subj = _clean(ex.get("subject"))
    msg  = _clean(ex.get("message"))
    if msg and msg.lower() != subj.lower():
        return (subj + " " + msg).strip() if subj else msg
    return subj or msg

def to_chat_codesn(ex):
    code   = _clean(ex.get("code"))
    target = _clean(ex.get("docstring"))
    user = ("Language: python\n"
            "Instruction: Explain clearly and concisely what this code does.\n"
            f"Code:\n```python\n{code}\n```")
    return {"text": f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n{ASSIST}{target}<|im_end|>\n"}

def to_chat_commit(ex):
    before = _clean(ex.get("old_code") or ex.get("old_contents"))
    after  = _clean(ex.get("new_code") or ex.get("new_contents"))
    target = _clean(ex.get("target")) or _synth_commit_target(ex)
    user = ("Language: python\n"
            "Instruction: Explain what changed between BEFORE and AFTER.\n"
            f"BEFORE:\n```python\n{before}\n```\nAFTER:\n```python\n{after}\n```")
    return {"text": f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n{ASSIST}{target}<|im_end|>\n"}

test_codesn = test_codesn_raw.map(to_chat_codesn, remove_columns=test_codesn_raw.column_names)
test_commit = test_commit_raw.map(to_chat_commit,  remove_columns=test_commit_raw.column_names)

# --- Tokenize & collate for eval (assistant-only labels) ---
MAX_LEN = 1536
RESPONSE_TAG = "<|im_start|>assistant\n"
ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

def tokenize_and_mask_factory(tok):
    def _f(ex):
        full = ex["text"]
        pref, _ = full.split(RESPONSE_TAG, 1)
        prompt_ids = tok(pref + RESPONSE_TAG, truncation=True, max_length=MAX_LEN, add_special_tokens=True)["input_ids"]
        enc = tok(full, truncation=True, max_length=MAX_LEN, add_special_tokens=True)
        ids, ams = enc["input_ids"], enc["attention_mask"]
        labels = ids.copy()
        cut = min(len(prompt_ids), len(labels))
        for i in range(cut): labels[i] = -100
        return {"input_ids": ids, "attention_mask": ams, "labels": labels}
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
    tot, n = 0.0, 0
    for b in loader:
        ids, ams, lbs = b["input_ids"].to(model.device), b["attention_mask"].to(model.device), b["labels"].to(model.device)
        logits = model(input_ids=ids, attention_mask=ams).logits
        sl, tl = logits[:, :-1, :], lbs[:, 1:]
        loss_tok = ce_loss(sl.reshape(-1, sl.size(-1)), tl.reshape(-1)).view(tl.size(0), -1)
        valid = (tl != -100).float()
        loss_per_ex = (loss_tok * valid).sum(1) / (valid.sum(1) + 1e-9)
        tot += float(loss_per_ex.sum()); n += loss_per_ex.numel()
    ce = tot / max(n, 1)
    ppl = math.exp(ce) if ce < 20 else float("inf")
    return ce, ppl

tnm  = tokenize_and_mask_factory(tok)
coll = pad_collate(tok)
codesn_tok = test_codesn.map(tnm, remove_columns=["text"])
commit_tok = test_commit.map(tnm, remove_columns=["text"])
codesn_ldr = DataLoader(codesn_tok, batch_size=2, shuffle=False, collate_fn=coll)
commit_ldr = DataLoader(commit_tok, batch_size=2, shuffle=False, collate_fn=coll)

codesn_ce, codesn_ppl = eval_ce_ppl(model, codesn_ldr)
commit_ce, commit_ppl = eval_ce_ppl(model, commit_ldr)

df = pd.DataFrame([
    {"dataset": "CodeSearchNet (TEST)", "test_ce": codesn_ce, "test_ppl": codesn_ppl},
    {"dataset": "CommitPackFT (TEST)",  "test_ce": commit_ce, "test_ppl": commit_ppl},
])

display(Markdown("## 📊 Cocktail-tuned model — evaluation on CodeSearchNet & CommitPackFT (TEST)"))
display(df.style.format({"test_ce": "{:.4f}", "test_ppl": "{:.2f}"}))

# Simple bar charts for quick visual comparison
import matplotlib.pyplot as plt
plt.figure(figsize=(7.2, 3.8))
plt.bar(df["dataset"], df["test_ce"])
plt.title("Test Cross-Entropy (lower is better)"); plt.ylabel("CE"); plt.grid(axis="y", alpha=.3)
plt.xticks(rotation=18, ha="right"); plt.tight_layout(); plt.show()

plt.figure(figsize=(7.2, 3.8))
plt.bar(df["dataset"], df["test_ppl"])
plt.title("Test Perplexity (lower is better)"); plt.ylabel("PPL"); plt.grid(axis="y", alpha=.3)
plt.xticks(rotation=18, ha="right"); plt.tight_layout(); plt.show()

# cleanup
del model
try: torch.cuda.empty_cache(); gc.collect()
except: pass
