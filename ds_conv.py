
import os
import json
from typing import Dict, List

ds_dir = "./dataset/"
ds_names = ds_dir + "names.json"
ds_out = ds_dir + "dataset.json"

names: Dict[str, Dict[str, str]] = None
with open(ds_names, 'r', encoding="utf-8") as fin:
    names = json.load(fin)

list = os.listdir(ds_dir)
out: List[Dict[str, str]] = []
for file in list:
    if not file.endswith(".schematic") and not file.endswith(".litematic"):
        continue
    index = file.split(".")[0]
    if index in names:
        item = names[index]
        out.append({
            "file": file,
            "title": item.get("name", item["description"])
        })

with open(ds_out, 'w', encoding = 'utf-8') as fout:
    json.dump(out, fout, ensure_ascii=False, indent=2)
