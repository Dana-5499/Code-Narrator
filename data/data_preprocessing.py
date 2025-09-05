!pip install -U "datasets<4.0" 
from datasets import load_dataset

# downloading each of the train, validation and test data set of code search net from hugging face
train_ds = load_dataset("code_search_net", split="train", trust_remote_code=True)
print("Train examples:", len(train_ds))
print("Fields:", train_ds.features)
print("Sample:", train_ds[0])

val_ds = load_dataset("code_search_net", split="validation", trust_remote_code=True)
print("Val examples:", len(val_ds))

test_ds = load_dataset("code_search_net", split="test", trust_remote_code=True)
print("Test examples:", len(test_ds))


# each sample contains the following fields:
# {'repository_name', 'func_path_in_repository','func_name', 'whole_func_string',
# 'language', 'func_code_string', 'func_code_tokens', 'func_documentation_string',
# 'func_documentation_tokens', 'split_name', 'func_code_url'}

# creating instruction-code-docstring pairs for each of the above data sets
def preprocess(example):
    return {
        "instruction": "Explain what the following code does.",
        "code":        example["func_code_string"],
        "docstring":   example["func_documentation_string"].strip(),
        "language":    example.get("language", "code")
    }

# apply preprocess function on each train data set and removing rest of fields
train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
val_ds = val_ds.map(preprocess, remove_columns=val_ds.column_names)
test_ds = test_ds.map(preprocess, remove_columns=test_ds.column_names)


# docstrings are usually of the following template:
# """ This is a reST style.
#:param param1: this is a first param
#:param param2: this is a second param
#:returns: this is a description of what is returned
#:raises keyError: raises an exception
#"""

# filtering any lines beginning with ':' to only keep human readable docstrings
def extract_summary(doc):
    summary_lines = []
    for raw in doc.splitlines():
        line = raw.strip()
        # if line is empty - we discard it
        if not line:
            break
        # if line begins with ':' - we discard it
        if line.startswith(":"):
            break
        summary_lines.append(line)
    # if docstring contains no human readable lines - we return none
    if not summary_lines:
        return None
    # joining multi line summaries
    return " ".join(summary_lines)

# filtering any docstrings which are not human readable:
# too short (less than 30 characters)
# must begin with an uppercase letter
# must end with a ./?/!
# doesn't start with see/refer referencing another source
def is_good_summary(s):
    if s is None:
        return False
    if len(s) < 30:
        return False
    if not s[0].isupper():
        return False
    if not s.endswith((".", "?", "!")):
        return False
    if s.startswith(("See","Refer")):
        return False
    return True

# applying summary extraction to each docstring
for ds in (train_ds, val_ds, test_ds):
    ds = ds.map(
        lambda example: {
            "instruction": example["instruction"],
            "code":        example["code"],
            "docstring":   extract_summary(example["docstring"]),
            "language":    example.get("language", "code")
        },
        remove_columns=ds.column_names  # discard the old docstring` field
    )
    # filtering out any examples where we got None or a bad summary
    ds = ds.filter(lambda example: is_good_summary(example["docstring"]))

    # reassigning the filtered docstring to their split group
    if ds.split == "train":
        train_ds = ds
    elif ds.split == "validation":
        val_ds = ds
    else:
        test_ds = ds
