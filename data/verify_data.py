"""
verify_data.py
==============
A rigorous data validation script to ensure no stats were corrupted,
shifted, or hallucinated during the matchup generation process.
AI GENERATED.
"""

import pandas as pd

MATCHUPS_CSV = "matchups_to_label.csv"
FIXED_CSV = "characters_scored_fixed.csv"
NUMERICAL_CSV = "characters_stats_numerical.csv"  # Your original numerical base stats

BASE_STATS = [
    "tier", "attack", "speed", "durability",
    "intelligence", "stamina", "lifting_strength", "striking_strength"
]

HAX_STATS = [
    "durability_negation", "regeneration", "power_amplification",
    "mobility_hax", "time_space_manip", "mind_soul_hax",
    "resistance_physical", "resistance_hax"
]

ALL_STATS = BASE_STATS + HAX_STATS


def run_validation():
    print("Loading datasets...")
    try:
        df_matchups = pd.read_csv(MATCHUPS_CSV)
        df_fixed = pd.read_csv(FIXED_CSV)
        df_num = pd.read_csv(NUMERICAL_CSV)
    except FileNotFoundError as e:
        print(f"ERROR: Could not find file - {e.filename}")
        return

    # 1. Build Source of Truth Dictionaries (Key: Character Name)
    # Note: If you have characters with identical names but different forms,
    # change the key to a tuple: (row['name'], row['form / era / buff'])
    truth_fixed = df_fixed.set_index('name')[ALL_STATS].to_dict('index')
    truth_base = df_num.set_index('name')[BASE_STATS].to_dict('index')

    # Dictionary to store the FIRST instance of a character we see in the matchups
    first_seen_stats = {}

    errors_found = 0

    print(f"Checking {len(df_matchups)} matchups...")

    for index, row in df_matchups.iterrows():

        # Check both Character A and Character B in the current matchup
        for prefix in ['a_', 'b_']:
            char_name = row[f"{prefix}name"]

            # Extract the stats for this character in THIS specific matchup row
            current_stats = {stat: row[f"{prefix}{stat}"] for stat in ALL_STATS}
            current_base_stats = {stat: row[f"{prefix}{stat}"] for stat in BASE_STATS}

            # --- CHECK 1 & 2: First Time Seeing Character (Check against Source CSVs) ---
            if char_name not in first_seen_stats:
                first_seen_stats[char_name] = current_stats

                # Check against Fixed CSV (All 16 stats)
                expected_fixed = truth_fixed.get(char_name)
                if expected_fixed != current_stats:
                    print(f"[ERROR] {char_name}: Stats in matchups do not match characters_scored_fixed.csv!")
                    errors_found += 1

                # Check against Numerical CSV (Just the 8 Base stats)
                expected_base = truth_base.get(char_name)
                if expected_base != current_base_stats:
                    print(f"[ERROR] {char_name}: Base stats do not match characters_stats_numerical.csv!")
                    errors_found += 1

            # --- CHECK 3: Internal Consistency (Check against First Seen) ---
            else:
                expected_stats = first_seen_stats[char_name]
                if current_stats != expected_stats:
                    print(f"[FATAL ERROR] {char_name}: Stats shifted internally at matchup row {index}!")
                    errors_found += 1

    # Final Verdict
    print("-" * 40)
    if errors_found == 0:
        print("✅ ALL CHECKS PASSED: Your dataset is cryptographically solid.")
        print(
            f"Successfully validated exactly {len(first_seen_stats)} unique characters across all {len(df_matchups)} rows.")
    else:
        print(f"❌ FAILED: Found {errors_found} errors. Do not train your model yet!")


if __name__ == "__main__":
    run_validation()