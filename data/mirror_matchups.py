"""
Doubles the training data by mirroring every matchup:
  A vs B -> winner  becomes  B vs A -> flipped winner
Eliminates positional bias and teaches the model relative stat relationships.
"""

import pandas as pd

INPUT = "matchups_winners_FIXED.csv"
OUTPUT = "matchups_winners_final.csv"

df = pd.read_csv(INPUT)

a_cols = [c for c in df.columns if c.startswith("a_")]
b_cols = [c for c in df.columns if c.startswith("b_")]

rename_map = {}
for c in a_cols:
    rename_map[c] = "b_" + c[2:]
for c in b_cols:
    rename_map[c] = "a_" + c[2:]

mirrored = df.rename(columns=rename_map).copy()
mirrored["winner"] = 1.0 - mirrored["winner"]

combined = pd.concat([df, mirrored], ignore_index=True)
combined = combined[df.columns]

print("Original rows:  " + str(len(df)))
print("Mirrored rows:  " + str(len(mirrored)))
print("Combined rows:  " + str(len(combined)))

combined.to_csv(OUTPUT, index=False)
print("Saved to " + OUTPUT)
