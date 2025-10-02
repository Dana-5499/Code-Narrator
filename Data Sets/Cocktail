# ===================== Cell 1: Mount Drive + paths =====================
# Purpose: Mount Google Drive, define source/target paths, sanity-check existence.
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

import os, random
SEED = 42                    # reproducibility anchor for shuffles/selects
random.seed(SEED)

# <<< EDIT THESE IF YOUR FOLDER NAMES DIFFER >>>
CODESN_PATH = "/content/drive/MyDrive/codesearchnet_compact_by_lang/python"  # CodeSearchNet (compact, python)
COMMIT_PATH = "/content/drive/MyDrive/commitpackft_compact_by_lang/python"   # CommitPackFT (compact, python)
OUT_DIR     = "/content/drive/MyDrive/cocktail_50_50_python"                 # output datasetdir for 50/50 mix

# sanity checks & output folder
assert os.path.exists(CODESN_PATH), f"Missing: {CODESN_PATH}"
assert os.path.exists(COMMIT_PATH), f"Missing: {COMMIT_PATH}"
os.makedirs(OUT_DIR, exist_ok=True)
print("OK: paths set.")

# ===================== Cell 2: Load datasets =====================
# Purpose: Load preprocessed per-language (python) compact datasets from disk.
# Expectation: Each has train/validation/test splits and consistent fields.
from datasets import load_from_disk, DatasetDict

ds_code   = load_from_disk(CODESN_PATH)   # expects train/validation/test
ds_commit = load_from_disk(COMMIT_PATH)   # expects train/validation/test

print("CodeSearchNet (compact/python):", {k: len(ds_code[k]) for k in ds_code})
print("CommitPackFT  (compact/python):", {k: len(ds_commit[k]) for k in ds_commit})

# quick schema peek (useful if earlier preprocessing differed)
print("\nCode columns:", ds_code["train"].column_names)
print("Commit columns:", ds_commit["train"].column_names)

# ===================== Cell 3: Uniform chat formatting (Qwen-style) =====================
# Purpose: Normalize both datasets to the same chat-ready 'text' column:
# <|im_start|>system ... <|im_end|>
# <|im_start|>user   ... <|im_end|>
# <|im_start|>assistant target <|im_end|>
#
# Notes:
# - Keep filtering conservative to preserve enough rows for the later caps.
# - Assistant-only loss masking later relies on the exact token: "<|im_start|>assistant\n"
import re

# --- CodeSearchNet -> (code, docstring) ---
SYS_CODE = ("You are a senior engineer. Summarize code for non-technical managers: "
            "plain English, 1–3 sentences, no jargon, no code, start with a verb.")

def to_chat_code(ex):
    code   = (ex.get("code") or "").strip()
    target = (ex.get("docstring") or "").strip()
    user = (f"Language: python\n"
            f"Instruction: Explain clearly and concisely what this code does.\n"
            f"Code:\n```python\n{code}\n```")
    return {"text":
        "<|im_start|>system\n" + SYS_CODE + "<|im_end|>\n" +
        "<|im_start|>user\n"   + user     + "<|im_end|>\n" +
        "<|im_start|>assistant\n" + target + "<|im_end|>\n"
    }

def good_code_row(ex):
    # Minimal signal gate: require some non-trivial code and docstring
    c = (ex.get("code") or "").strip()
    d = (ex.get("docstring") or "").strip()
    return (len(c) >= 10) and (len(d) >= 10)

# --- CommitPackFT -> (old_code/new_code, target or subject/message fallback) ---
SYS_COMMIT = ("You are a senior engineer. Explain the functional change between two code revisions "
              "in plain English, 1–3 sentences, no jargon, no code, start with a verb.")

def _synth_target(ex):
    # When 'target' is missing, derive from subject/message (dedup & whitespace-normalize)
    subj = (ex.get("subject") or "").strip()
    msg  = (ex.get("message") or "").strip()
    if msg and msg.lower() != subj.lower():
        combined = (subj + " " + msg).strip() if subj else msg
    else:
        combined = subj or msg
    combined = re.sub(r"\s+", " ", combined or "").strip()
    return combined if combined else None

def good_commit_row(ex):
    # Require both BEFORE and AFTER, plus a non-trivial target
    before = (ex.get("old_code") or ex.get("old_contents") or "").strip()
    after  = (ex.get("new_code") or ex.get("new_contents") or "").strip()
    tgt    = (ex.get("target") or _synth_target(ex) or "").strip()
    return (len(before) >= 10) and (len(after) >= 10) and (len(tgt) >= 10)

def to_chat_commit(ex):
    before = (ex.get("old_code") or ex.get("old_contents") or "").strip()
    after  = (ex.get("new_code") or ex.get("new_contents") or "").strip()
    target = (ex.get("target") or _synth_target(ex) or "").strip()
    user = ("Language: code\n"
            "Instruction: Explain clearly and concisely what changed between BEFORE and AFTER.\n"
            "BEFORE:\n```diff\n" + before + "\n```\n\n"
            "AFTER:\n```diff\n"  + after  + "\n```")
    return {"text":
        "<|im_start|>system\n" + SYS_COMMIT + "<|im_end|>\n" +
        "<|im_start|>user\n"   + user       + "<|im_end|>\n" +
        "<|im_start|>assistant\n" + target  + "<|im_end|>\n"
    }

# Filter each split conservatively, then map to 'text'
code_train = ds_code["train"].filter(good_code_row).map(to_chat_code, remove_columns=ds_code["train"].column_names)
code_val   = ds_code["validation"].filter(good_code_row).map(to_chat_code, remove_columns=ds_code["validation"].column_names)
code_test  = ds_code["test"].filter(good_code_row).map(to_chat_code, remove_columns=ds_code["test"].column_names)

commit_train = ds_commit["train"].filter(good_commit_row).map(to_chat_commit, remove_columns=ds_commit["train"].column_names)
commit_val   = ds_commit["validation"].filter(good_commit_row).map(to_chat_commit, remove_columns=ds_commit["validation"].column_names)
commit_test  = ds_commit["test"].filter(good_commit_row).map(to_chat_commit, remove_columns=ds_commit["test"].column_names)

print("After formatting ->")
print("Code:",   { "train": len(code_train),   "val": len(code_val),   "test": len(code_test) })
print("Commit:", { "train": len(commit_train), "val": len(commit_val), "test": len(commit_test) })

# ===================== Cell 4: Build the 50/50 cocktail (caps) =====================
# Purpose: Create a 50/50 mixture by concatenating equal-sized samples (with caps) from each source.
# Targets are per-split totals; we take half from each dataset. If a split is smaller than the cap,
# we simply take as many as available (no upsampling/replacement).
from datasets import concatenate_datasets, DatasetDict

# Targets (TOTAL per split)
TOTALS = {"train": 2100, "validation": 300, "test": 600}
HALF   = {k: v // 2 for k, v in TOTALS.items()}  # 1050/150/300

def take_n(ds, n, seed=SEED):
    # Shuffle (seeded) and select up to n rows, clamped by dataset size
    n_keep = min(n, len(ds))
    return ds.shuffle(seed=seed).select(range(n_keep))

cocktail_train = concatenate_datasets([
    take_n(code_train,   HALF["train"]),
    take_n(commit_train, HALF["train"])
]).shuffle(seed=SEED)

cocktail_val = concatenate_datasets([
    take_n(code_val,   HALF["validation"]),
    take_n(commit_val, HALF["validation"])
]).shuffle(seed=SEED)

cocktail_test = concatenate_datasets([
    take_n(code_test,   HALF["test"]),
    take_n(commit_test, HALF["test"])
]).shuffle(seed=SEED)

cocktail = DatasetDict({
    "train": cocktail_train,
    "validation": cocktail_val,
    "test": cocktail_test
})

print("Cocktail sizes (requested totals):", TOTALS)
print("Actual ->", {k: len(cocktail[k]) for k in cocktail})

# ===================== Cell 5: Save + quick sanity check =====================
# Purpose: Save the combined dataset to Drive; reload to verify contents; print a few samples.
cocktail.save_to_disk(OUT_DIR)
print("Saved cocktail to:", OUT_DIR)

# Reload to be sure persisted content is correct
reloaded = load_from_disk(OUT_DIR)
print("Reloaded sizes:", {k: len(reloaded[k]) for k in reloaded})

# Peek a sample (first ~1000 chars of the chat-formatted 'text')
print("\nSample (train[0]['text']):\n")
print(reloaded["train"][0]["text"][:1000])

# Print a few more samples to spot-check variety and structure
print("\nAdditional Samples:\n")
for i in range(1, 5):
    print(f"Sample {i+1}:\n")
    print(reloaded["train"][i]["text"][:1000])
    print("-" * 50)
