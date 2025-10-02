# ======================================================
# CommitPackFT preprocessing pipeline
#   1) Load multiple language configs from HF (CommitPackFT).
#   2) Stratified split into train/val/test (70/10/20).
#   3) Clean & normalize commit messages (subject+message).
#   4) Filter commits by readability/quality.
#   5) Save processed dataset to Google Drive.
#   6) Build per-language splits and compact subsets (2,100/300/600).
# NOTE: Code is unchanged — only explanatory comments were added.
# ======================================================

# --- Install HuggingFace datasets library (pinned <4.0 for compatibility) ---
!pip install -q "datasets<4.0"
from datasets import load_dataset, concatenate_datasets

# Languages to include from CommitPackFT
LANGS = ["python", "java", "javascript", "go", "ruby", "php"]

# --- Load datasets per language and tag them explicitly ---
parts = []
for lang in LANGS:
    print(f"Loading {lang}...")
    ds_lang = load_dataset("bigcode/commitpackft", lang, split="train")
    # Add an explicit 'language' column (sometimes missing in the raw data)
    ds_lang = ds_lang.map(lambda e: {"language": lang})
    parts.append(ds_lang)

# --- Combine into one dataset across all selected languages ---
ds = concatenate_datasets(parts)

print("\nFinal dataset")
print("Total rows:", len(ds))
print("Fields:", ds.features)
print("Sample:", ds[0])

# ======================================================
# Split dataset into Train / Validation / Test
# ======================================================
from datasets import ClassLabel, DatasetDict

# Encode "language" column as a ClassLabel (categorical type)
langs = sorted(set(ds["language"]))
classlabel = ClassLabel(num_classes=len(langs), names=langs)
ds = ds.cast_column("language", classlabel)

# Stratified split:
#   70% train
#   10% validation
#   20% test
split_1 = ds.train_test_split(test_size=0.30, seed=42, stratify_by_column="language")
split_2 = split_1["test"].train_test_split(test_size=2/3, seed=42, stratify_by_column="language")

splits = DatasetDict({
    "train": split_1["train"],
    "validation": split_2["train"],
    "test": split_2["test"]
})

print("Train size:", len(splits["train"]))
print("Validation size:", len(splits["validation"]))
print("Test size:", len(splits["test"]))

# ======================================================
# Language distribution diagnostic
# ======================================================
from collections import Counter

def print_language_distribution(ds, split_name="dataset"):
    feat = ds.features["language"]
    if isinstance(feat, ClassLabel):
        langs = [feat.int2str(i) for i in ds["language"]]  # decode indices
    else:
        langs = ds["language"]

    counts = Counter(langs)
    print(f"\nLanguage distribution in the {split_name} split:")
    for lang, count in sorted(counts.items(), key=lambda x: x[0].lower()):
        print(f"{lang:12s} {count}")

print_language_distribution(splits["train"], "train")
print_language_distribution(splits["validation"], "validation")
print_language_distribution(splits["test"], "test")

# ======================================================
# Inspect random commit samples (for sanity checks)
# ======================================================
feat = splits["train"].features["language"]

def print_random_samples(ds, n=3, seed=42):
    sample_ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    for i, ex in enumerate(sample_ds, 1):
        lang_str = feat.int2str(ex["language"]) if isinstance(ex["language"], int) else ex["language"]
        print(f"\n{'='*40} SAMPLE {i} {'='*40}")
        print("Language:", lang_str)
        print("\n--- Subject ---")
        print(ex.get("subject", ""))
        print("\n--- Message ---")
        print(ex.get("message", ""))
        print("\n--- Old contents ---")
        print((ex.get("old_contents") or "")[:800])
        print("\n--- New contents ---")
        print((ex.get("new_contents") or "")[:800])
        print("\n" + "="*90)

print_random_samples(splits["train"], n=3)

# ======================================================
# Preprocess: keep only relevant fields
# ======================================================
def preprocess_commit(example):
    return {
        "instruction": "Explain, in plain English, what functionality changed between these two commits.",
        "old_code":   (example.get("old_contents") or "").strip(),
        "new_code":   (example.get("new_contents") or "").strip(),
        "language":   example.get("language", example.get("lang", "code")),
        "message":    (example.get("message") or "").strip(),
        "subject":    (example.get("subject") or "").strip(),
    }

for split_name in ["train", "validation", "test"]:
    splits[split_name] = splits[split_name].map(
        preprocess_commit,
        remove_columns=splits[split_name].column_names
    )

print(splits)
print("Train sample:", splits["train"][0])

# ======================================================
# Filtering: clean commit messages (subject+message)
# ======================================================
import re

def extract_commit_message(subject, message):
    # Prefer subject; append message if it adds new info
    subj = (subject or "").strip()
    msg  = (message or "").strip()
    if msg and msg.lower() != subj.lower():
        combined = subj + "\n" + msg if subj else msg
    else:
        combined = subj or msg
    # Normalize whitespace
    combined = re.sub(r"\s+", " ", combined).strip()
    return combined if combined else None

def is_good_commit(msg):
    if msg is None: return False
    if len(msg) < 20: return False         # too short
    if not msg[0].isupper(): return False  # must start with uppercase
    if not msg.endswith((".", "?", "!")): return False  # must end with punctuation
    # discard boilerplate commits
    bad_starts = ("Merge branch", "Merge pull request", "WIP", "Temp", "Test", 
                  "Update README", "Bump version", "Release")
    if msg.startswith(bad_starts): return False
    return True

def preprocess_and_filter(ds):
    ds = ds.map(lambda ex: {
        **ex,
        "target": extract_commit_message(ex.get("subject"), ex.get("message"))
    })
    return ds.filter(lambda ex: is_good_commit(ex["target"]))

splits["train"] = preprocess_and_filter(splits["train"])
splits["validation"] = preprocess_and_filter(splits["validation"])
splits["test"] = preprocess_and_filter(splits["test"])

print("Train size:", len(splits["train"]))
print("Validation size:", len(splits["validation"]))
print("Test size:", len(splits["test"]))

# ======================================================
# Save processed dataset to Google Drive
# ======================================================
from google.colab import drive
import os

drive.mount('/content/drive', force_remount=True)
save_dir = "/content/drive/MyDrive/commitpackft_processed"
os.makedirs(save_dir, exist_ok=True)
ds = DatasetDict(splits)
ds.save_to_disk(save_dir)
print("Saved processed dataset to", save_dir)

print("Train size:", len(ds["train"]))
print("Validation size:", len(ds["validation"]))
print("Test size:", len(ds["test"]))

# ======================================================
# Split per language
# ======================================================
from collections import defaultdict

def split_by_language_commits(splits):
    train_ds, val_ds, test_ds = splits["train"], splits["validation"], splits["test"]
    lang_splits = defaultdict(dict)

    feat = train_ds.features["language"]
    all_langs = set(train_ds["language"]) | set(val_ds["language"]) | set(test_ds["language"])

    for lang in all_langs:
        lang_splits[lang]["train"] = train_ds.filter(lambda e: e["language"] == lang)
        lang_splits[lang]["validation"] = val_ds.filter(lambda e: e["language"] == lang)
        lang_splits[lang]["test"] = test_ds.filter(lambda e: e["language"] == lang)

    return {lang: DatasetDict(splits) for lang, splits in lang_splits.items()}

lang_datasets = split_by_language_commits(splits)

for lang, dsets in lang_datasets.items():
    print(f"\n=== {lang.upper()} ===")
    for split_name, ds in dsets.items():
        print(f"{split_name}: {len(ds)} samples")

# ======================================================
# Compact each language split (2100/300/600 max)
# ======================================================
import hashlib, random
from datasets import DatasetDict

TARGETS = {"train": 2_100, "validation": 300, "test": 600}
random.seed(42)

def _get_before_after(ex):
    return ex.get("old_code") or "", ex.get("new_code") or ""

def good_change_length(ex, min_lines=3, max_lines=350):
    before, after = _get_before_after(ex)
    n = (before.count("\n") + 1) + (after.count("\n") + 1)
    return (min_lines <= n <= max_lines)

def add_change_hash(ex):
    before, after = _get_before_after(ex)
    ex["__change_hash"] = hashlib.md5(
        (before.strip() + "\n<SEP>\n" + after.strip()).encode("utf-8","ignore")
    ).hexdigest()
    return ex

def unique_by_change_hash():
    seen = set()
    def _f(e):
        h = e["__change_hash"]
        if h in seen: return False
        seen.add(h)
        return True
    return _f

def reduce_and_sample(ds, target, seed=42):
    if len(ds) == 0: return ds
    ds = ds.filter(good_change_length, desc="len filter")
    ds = ds.map(add_change_hash, desc="hashing")
    ds = ds.filter(unique_by_change_hash(), desc="dedup")
    keep = min(target, len(ds))
    ds_small = ds.shuffle(seed=seed).select(range(keep))
    if "__change_hash" in ds_small.column_names:
        ds_small = ds_small.remove_columns(["__change_hash"])
    return ds_small

per_lang_compact = {}
for lang, splits in lang_datasets.items():
    compact_train = reduce_and_sample(splits["train"],      TARGETS["train"])
    compact_val   = reduce_and_sample(splits["validation"], TARGETS["validation"])
    compact_test  = reduce_and_sample(splits["test"],       TARGETS["test"])
    per_lang_compact[lang] = DatasetDict({
        "train": compact_train, "validation": compact_val, "test": compact_test
    })

# Print counts
for lang, dsets in per_lang_compact.items():
    print(f"\n=== {str(lang).upper()} (compact) ===")
    for split_name, ds_ in dsets.items():
        print(f"{split_name}: {len(ds_)} samples")

# Save compact datasets per language
base_dir = "/content/drive/MyDrive/commitpackft_compact_by_lang"
os.makedirs(base_dir, exist_ok=True)
for lang, dsets in per_lang_compact.items():
    safe = re.sub(r"[^a-zA-Z0-9_+-]+", "_", str(lang).lower())
    outdir = os.path.join(base_dir, safe)
    dsets.save_to_disk(outdir)
    print(f"Saved {lang} compact dataset to {outdir}")
