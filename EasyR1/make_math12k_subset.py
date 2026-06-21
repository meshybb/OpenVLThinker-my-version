from datasets import load_dataset
import json
import random

random.seed(1)

ds_train = load_dataset("hiyouga/math12k", split="train")
ds_test = load_dataset("hiyouga/math12k", split="test")

def convert(ds, n, prefix):
    rows = []
    for i, ex in enumerate(ds.select(range(min(n, len(ds))))):
        rows.append({
            "problem": ex["problem"],
            "answer": ex["answer"],
            "problem_type": "math",
            "data_type": "text",
            "task_type": "math_reasoning",
            "images": [],
            "videos": [],
            "problem_id": f"{prefix}_{i}"
        })
    return rows

train_rows = convert(ds_train, 64, "math12k_train")
val_rows = convert(ds_test, 50, "math12k_val")

with open("data_small/math12k_train_64.json", "w", encoding="utf-8") as f:
    json.dump(train_rows, f, ensure_ascii=False, indent=2)

with open("data_small/math12k_val_50.json", "w", encoding="utf-8") as f:
    json.dump(val_rows, f, ensure_ascii=False, indent=2)

print("created train:", len(train_rows))
print("created val:", len(val_rows))
print(train_rows[0])
