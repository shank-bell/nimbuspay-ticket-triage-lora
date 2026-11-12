"""Build the two transfer zips for the college PC into college_transfer/ (git-ignored).

  nimbus_code.zip         code only (safe to share)
  candidate_pack_data.zip company train/dev data + score.py + SCHEMA.md (PRIVATE; test inputs are deliberately left out)
"""
import os, zipfile

ROOT = "E:/Fine_Tuneing-Assigh/"
OUT = ROOT + "college_transfer/"
PACK = ROOT + "training_assessment/candidate_pack/"
os.makedirs(OUT, exist_ok=True)

code = [("college/run_college.py", "nimbus_code/run_college.py"), ("college/README_COLLEGE.md", "nimbus_code/README_COLLEGE.md"), ("college/MASTER_PROMPT.md", "nimbus_code/MASTER_PROMPT.md"),
        ("src/nimbus_train.py", "nimbus_code/src/nimbus_train.py"), ("audit/nimbus_lib.py", "nimbus_code/audit/nimbus_lib.py"),
        ("audit/nimbus_clean.py", "nimbus_code/audit/nimbus_clean.py")]
data = [("data/train.jsonl", "candidate_pack/data/train.jsonl"), ("data/dev.jsonl", "candidate_pack/data/dev.jsonl"),
        ("score.py", "candidate_pack/score.py"), ("SCHEMA.md", "candidate_pack/SCHEMA.md")]

for zname, items, base in (("nimbus_code.zip", code, ROOT), ("candidate_pack_data.zip", data, PACK)):
    with zipfile.ZipFile(OUT + zname, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in items:
            z.write(base + src, arc)
    print(f"{zname}: {os.path.getsize(OUT + zname) / 1024:,.0f} KB")
    with zipfile.ZipFile(OUT + zname) as z:
        for i in z.infolist():
            print(f"    {i.file_size:>10,}  {i.filename}")
