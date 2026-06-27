"""
generate_matchups.py
====================
Reads the fixed dataset and generates the FULL XGBoost training matrix,
but only in one direction (A vs B) to save API costs (2,346 rows).
"""

import pandas as pd
from itertools import combinations
import random

INPUT_CSV = "characters_scored_fixed.csv"
OUTPUT_CSV = "matchups_to_label.csv"

# All the numerical columns your XGBoost model will train on
ALL_FEATURES = [
    "tier", "attack", "speed", "durability",
    "intelligence", "stamina", "lifting_strength", "striking_strength",
    "durability_negation", "regeneration", "power_amplification",
    "mobility_hax", "time_space_manip", "mind_soul_hax",
    "resistance_physical", "resistance_hax"
]

def generate_full_training_matrix():
    df = pd.read_csv(INPUT_CSV)

    # Generate unique pairs (2,346 total)
    pairs = list(combinations(df.index, 2))
    random.shuffle(pairs)

    rows = []
    for idx_a, idx_b in pairs:
        a = df.loc[idx_a]
        b = df.loc[idx_b]

        row = {}

        # --- Char A Identifiers & Stats ---
        row["a_name"]  = a["name"]
        row["a_form"]  = a["form / era / buff"]
        row["a_anime"] = a["anime"]
        for c in ALL_FEATURES:
            row[f"a_{c}"] = a[c]

        # --- Char B Identifiers & Stats ---
        row["b_name"]  = b["name"]
        row["b_form"]  = b["form / era / buff"]
        row["b_anime"] = b["anime"]
        for c in ALL_FEATURES:
            row[f"b_{c}"] = b[c]

        # --- The Label (To be filled by Haiku) ---
        row["winner"] = None

        rows.append(row)

    matchups_df = pd.DataFrame(rows)
    matchups_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Successfully generated {len(matchups_df)} one-way matchups.")
    print(f"File '{OUTPUT_CSV}' now contains all stats and is ready for LLM labeling!")

if __name__ == "__main__":
    generate_full_training_matrix()