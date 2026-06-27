"""
Anime Versus ML - Synthetic Data Generator
Module: src/llm_request.py
Description: Interfaces with the Claude API to generate synthetic community-consensus
             win/loss labels for generated matchups based strictly on canon lore,
             ignoring raw statistical inputs.
Author: Aiden J. Tejada
Date: 2026-05-22
License: MIT License
"""

__version__ = "0.1.0"
__author__ = "Aiden J. Tejada"
__license__ = "MIT License"


import pandas as pd
import os
import time
import dotenv
import json
import anthropic
from prompts import WINNER_PROMPT, WINNER_SCHEMA, ABILITY_PROMPT, ABILITY_SCHEMA, SYSTEM_PROMPT_ABILITY, SYSTEM_PROMPT_WINNER



# config
dotenv.load_dotenv()
claude_key = os.getenv("CLAUDE_API_KEY")
client = anthropic.Anthropic(api_key=claude_key)
matchups_path = "../data/matchups_to_label.csv"
output_path = "../data/matchups_winners.csv"
character_path = "../data/characters_stats_numerical.csv"
df_matchups = pd.read_csv(matchups_path)
df_characters = pd.read_csv(character_path)



def get_ability_scores():
    """
    Sends requests to Claude API to score different abilites.
    This is to be ran once, and edited manually in case any mistakes. P.S (There were lots)
    Saves file to ../data/characters_scored_final.csv
    """
    for i, row in df_characters.iterrows():
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929", # smarter model here, more expensive but.. i need GOOD synthetic data.
            max_tokens=512,
            system=SYSTEM_PROMPT_ABILITY,
            messages=[
                {"role": "user", "content": ABILITY_PROMPT.format(name=row["name"], form=row["form / era / buff"], anime=row["anime"])}
            ],
            tools=[ABILITY_SCHEMA],
            tool_choice={"type": "tool", "name": "ability_scores"}
        )
        result = response.content[0].input
        ABILITY_COLS = [
            "durability_negation", "regeneration", "power_amplification",
            "mobility_hax", "time_space_manip", "mind_soul_hax",
            "resistance_physical", "resistance_hax"
        ]
        #validation, making sure the right numbers get put in the right place, if not, flag.
        valid = all(
            col in result and isinstance(result[col], (int, float)) and 1.0 <= result[col] <= 10.0
            for col in ABILITY_COLS
        )

        if valid:
            for col in ABILITY_COLS:
                df_characters.at[i, col] = result[col]
            print(f"Scored: {row['name']} — {result}")
        else:
            print(f"[INVALID] {row['name']} — {result}")

        time.sleep(60 / 49)
    df_characters.to_csv("../data/characters_scored_final.csv", index=False)
def get_matchup_winners():
    """
    Uses an LLM to decide the winnder based on lore and domain knowledge.
    This allows the ML model to determine its own weights instead of letting the LLM be deterministic with my data.
    Saves to ../data/matchups_winners.csv
    All winners verified in batches of 100 copy pasting into multiple chatbots (Claude, ChatGPT, etc) and cross-referencing / marking incorrect winners
    All marked winners are then changed manually. Just to insure my data is 100% accurate.
    """
    for i, row in df_matchups.iterrows():
        if pd.notna(row["winner"]) and str(row["winner"]).strip() != "": #skips if row already has a winner and allows me to resume if theres an error, internet goes out or i run out of money LOL
            continue
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001", #cheap model, smart enough to determine a winner
                max_tokens=512,
                system=SYSTEM_PROMPT_WINNER,
                messages=[
                    {"role": "user", "content": WINNER_PROMPT.format(
                    a_name=row["a_name"], a_form=row["a_form"], a_anime=row["a_anime"],
                    b_name=row["b_name"], b_form=row["b_form"], b_anime=row["b_anime"] # only sending the names, forms, and anime
                                    )}],
                tools=[WINNER_SCHEMA],
                tool_choice={"type": "tool", "name": "matchup_winner"}

            )
            result = response.content[0].input
            df_matchups.at[i, "winner"] = result["winner"]
            print(f"Matchup {i+1}: {row['a_name']} vs {row['b_name']} — {result['winner']}")
            df_matchups.to_csv("../data/matchups_winners.csv", index=False) # saves after each loop as to not lose progress / waste tokens
            time.sleep(60 / 49)
        except Exception as e:
            print(f"Error processing matchup {i+1}: {e}")
            df_matchups.to_csv("../data/matchups_winners.csv", index=False) # saves in case of a crash
            break
    print("Done.")

if __name__ == "__main__":
    # get_ability_scores()  # archival: Executed once during data generation not needed anymore
    get_matchup_winners()
