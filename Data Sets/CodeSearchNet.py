# ================================
# Setup & Data Build Script (CodeSearchNet → cleaned → per-language compact)
# IMPORTANT:
#  • Logic is UNCHANGED — only comments were added for clarity.
#  • Pins datasets<4.0 for compatibility with Colab/older stacks.
#  • Downloads the full CodeSearchNet splits, cleans docstrings, filters code,
#    deduplicates by code hash, then saves both a processed global dataset and
#    compact per-language subsets (2,100/300/600).
# ================================

!pip -q install -U "datasets<4.0"

from google.colab import drive
drive.mount("/content/drive", force_remount=True)

from datasets import load_dataset, DatasetDict
from collections import Counter, defaultdict
import os, re, hashlib, random

random.seed(42)  # reproducibility for shuffles/selections

# ========== 1) Download CodeSearchNet ==========
# Grabs full train/val/test splits (can be large). trust_remote_code=True is
# required by some HF datasets to run their custom prepare scripts.
train_ds = load_dataset("code_search_net", split="train", trust_remote_code=True)
val_ds   = load_dataset("code_search_net", split="validation", trust_remote_code=True)
test_ds  = load_dataset("code_search_net", split="test", trust_remote_code=True)  # <-- complete line

print("Lang dist (train):", Counter(train_ds["language"]).most_common(10))

# ========== 2) Basic preprocess ==========
# Normalize the raw schema to a stable, compact set of fields and add a generic instruction.
def preprocess(example):
    return {
        "instruction": "Explain what the following code does.",
        "code":        example["func_code_string"],
        "docstring":   example["func_documentation_string"] or "",
        "language":    example.get("language", "code")
    }

# Apply the lightweight projection (keeps only the columns we care about).
train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(preprocess,   remove_columns=val_ds.column_names)
test_ds  = test_ds.map(preprocess,  remove_columns=test_ds.column_names)

# Clean docstrings
# Extract only the leading summary (first non-empty paragraph, stop on Sphinx-style ":" lines).
def extract_summary(doc):
    summary_lines = []
    for raw in (doc or "").splitlines():
        line = raw.strip()
        if not line: break              # stop at first blank line (end of summary block)
        if line.startswith(":"): break  # stop at Sphinx param/returns sections
        summary_lines.append(line)
    if not summary_lines:
        return None
    return " ".join(summary_lines)

# Basic quality gate for summaries (length, capitalization, punctuation, avoid weak “See/Refer”).
def is_good_summary(s):
    if s is None: return False
    if len(s) < 30: return False
    if not s[0].isupper(): return False
    if not s.endswith((".", "?", "!")): return False
    if s.startswith(("See","Refer")): return False
    return True

# Apply summary extraction + quality filter per split.
def clean_split(ds):
    ds = ds.map(lambda ex: {
        "instruction": ex["instruction"],
        "code": ex["code"],
        "docstring": extract_summary(ex["docstring"]),
        "language": ex.get("language", "code")
    })
    ds = ds.filter(lambda ex: is_good_summary(ex["docstring"]))
    return ds

train_ds = clean_split(train_ds)
val_ds   = clean_split(val_ds)
test_ds  = clean_split(test_ds)

print("Sizes ->", { "train": len(train_ds), "validation": len(val_ds), "test": len(test_ds) })

# ========== 3) Split per-language ==========
# Builds a DatasetDict per language so we can later save per-language compact sets.
def split_by_language(train_ds, val_ds, test_ds):
    langs = set(train_ds["language"]) | set(val_ds["language"]) | set(test_ds["language"])
    lang_splits = {}
    for lang in langs:
        lang_splits[lang] = DatasetDict({
            "train":      train_ds.filter(lambda e, l=lang: e["language"] == l),
            "validation": val_ds.filter(lambda e, l=lang: e["language"] == l),
            "test":       test_ds.filter(lambda e, l=lang: e["language"] == l),
        })
    return lang_splits

lang_datasets = split_by_language(train_ds, val_ds, test_ds)

# ========== 4) Make compact per-language subsets ==========
# Target sizes match your project spec (2,100 train / 300 val / 600 test).
TARGETS = {"train": 2_100, "validation": 300, "test": 600}

# Filters out extremely short/long code blocks (by line count).
def good_code_length(code, min_lines=3, max_lines=350):
    if not code: return False
    n = code.count("\n") + 1
    return (min_lines <= n <= max_lines)

# Provide a stable MD5 hash on normalized code string to deduplicate exact duplicates.
def add_code_hash(ex):
    ex["__code_hash"] = hashlib.md5((ex["code"] or "").strip().encode("utf-8","ignore")).hexdigest()
    return ex

# Keep only first occurrence per code hash.
def unique_by_hash():
    seen = set()
    def _f(e):
        h = e["__code_hash"]
        if h in seen: return False
        seen.add(h)
        return True
    return _f

# Reduce: filter by length → hash → dedup → shuffle → sample K
def reduce_and_sample(ds, target, seed=42):
    ds = ds.filter(lambda e: good_code_length(e["code"]), desc="len filter")
    ds = ds.map(add_code_hash, desc="hashing")
    ds = ds.filter(unique_by_hash(), desc="dedup")
    keep = min(target, len(ds))
    ds_small = ds.shuffle(seed=seed).select(range(keep))
    if "__code_hash" in ds_small.column_names:
        ds_small = ds_small.remove_columns(["__code_hash"])
    return ds_small

# Build compact splits per language.
per_lang_compact = {}
for lang, splits in lang_datasets.items():
    compact_train = reduce_and_sample(splits["train"],      TARGETS["train"])
    compact_val   = reduce_and_sample(splits["validation"], TARGETS["validation"])
    compact_test  = reduce_and_sample(splits["test"],       TARGETS["test"])
    per_lang_compact[lang] = DatasetDict({
        "train": compact_train, "validation": compact_val, "test": compact_test
    })

# ========== 5) SAVE (GLOBAL + PER-LANG) ==========
# 5a) Save the global processed dataset (all languages combined) — full size after cleaning
processed_dir = "/content/drive/MyDrive/codesearchnet_processedguy"
os.makedirs(processed_dir, exist_ok=True)
DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds}).save_to_disk(processed_dir)
print("Saved processed dataset to", processed_dir)

# 5b) Save per-language compact datasets (small, balanced slices for quick training)
compact_base = "/content/drive/MyDrive/codesearchnet_compact_by_langguy"
os.makedirs(compact_base, exist_ok=True)
for lang, dsets in per_lang_compact.items():
    safe = re.sub(r"[^a-zA-Z0-9_+-]+", "_", lang.lower())
    outdir = os.path.join(compact_base, safe)
    dsets.save_to_disk(outdir)
    print(f"Saved {lang} compact dataset to {outdir}")

# 5c) Verify the expected python/ folder exists and list contents for sanity.
print("\nCompact parent contents:", os.listdir(compact_base))
py_dir = os.path.join(compact_base, "python")
print("python exists?", os.path.exists(py_dir))
if os.path.exists(py_dir):
    print("python contents:", os.listdir(py_dir))
