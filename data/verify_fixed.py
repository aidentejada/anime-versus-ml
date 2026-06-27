import pandas as pd
import re

df = pd.read_csv("matchups_winners_FIXED.csv")
with open("matchups_winners_to_fix.txt", "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]

p = re.compile(
    r"\[(\d+)\]\s+(.+?)\s+\((.+?)\)\s+vs\s+(.+?)\s+\((.+?)\)\s*\|\s*([\d.]+)\s*->\s*([\d.]+)"
)

total = 0
ok = 0
bad = []

for line in lines:
    m = p.match(line)
    if not m:
        continue
    total += 1
    idx = int(m.group(1))
    a_name = m.group(2).strip()
    b_name = m.group(4).strip()
    new_val = float(m.group(7))

    row = df.iloc[idx]
    cur = float(row["winner"])
    name_ok = row["a_name"] == a_name and row["b_name"] == b_name
    val_ok = cur == new_val

    if name_ok and val_ok:
        ok += 1
    else:
        issues = []
        if not name_ok:
            issues.append("NAME MISMATCH: csv=" + row["a_name"] + " vs " + row["b_name"])
        if not val_ok:
            issues.append("VALUE: got=" + str(cur) + " expected=" + str(new_val))
        bad.append("[" + str(idx) + "] " + a_name + " vs " + b_name + " -- " + ", ".join(issues))

print("Checked: " + str(total))
print("Correct: " + str(ok))
print("Failed:  " + str(len(bad)))
if bad:
    for b in bad:
        print(b)
else:
    print()
    print("ALL FIXES CONFIRMED -- FIXED CSV has the correct new winner value for every entry.")
