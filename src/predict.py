"""
Anime Versus ML - Inference Engine
Module: src/predict.py
Description: Loads the serialized XGBoost model and provides a clean function to
             evaluate new, unseen 1v1 character matchups based on input vectors.
Author: Aiden J. Tejada
Date: 2026-05-22
License: MIT License
"""

__version__ = "0.1.0"
__author__ = "Aiden J. Tejada"
__license__ = "MIT License"


import xgboost as xgb
import pandas as pd
import shap
import os
import dotenv
from openai import OpenAI
from prompts import SYSTEM_PROMPT_EXPLAIN, EXPLAIN_PROMPT
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

#fast API setup
app = FastAPI()
app.mount("/imgs", StaticFiles(directory="../data/imgs"), name="imgs")

@app.get("/")
def home():
    return FileResponse("../frontend/index.html")

dotenv.load_dotenv()
llm = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))

characters = pd.read_csv("../data/characters_scored_fixed_NOTES.csv")
model = xgb.XGBClassifier()
model.load_model("../models/xgb_model.json")
explainer = shap.TreeExplainer(model)

STAT_COLS = [
    "tier", "attack", "speed", "durability", "intelligence", "stamina",
    "lifting_strength", "striking_strength", "durability_negation",
    "regeneration", "power_amplification", "mobility_hax",
    "time_space_manip", "mind_soul_hax", "resistance_physical", "resistance_hax"
]

DROP_COLS = ["a_name", "a_form", "a_anime", "b_name", "b_form", "b_anime"]

@app.get("/chars")
def list_characters():
    # columns the frontend actually needs to render the selection grid + stat panel
    cols = ["name", "form / era / buff", "anime", "image_links", "notes"] + STAT_COLS
    subset = characters[cols].copy()
    subset.insert(0, "id", subset.index)          # row number becomes a stable "id" field
    subset = subset.astype(object).where(pd.notna(subset), None)  # object dtype so None sticks (NaN invalid in JSON)
    return subset.to_dict(orient="records")        # DataFrame -> list of dicts -> JSON


def print_character_list():
    col_w = 35
    print("\nAvailable Characters:")
    print("|" + "-" * col_w + "|" + "-" * col_w + "|" + "-" * col_w + "|")
    rows = list(characters.iterrows())
    for i in range(0, len(rows), 3):
        parts = []
        for j in range(3):
            if i + j < len(rows):
                idx, row = rows[i + j]
                cell = f" {idx + 1:>3}. {row['name'][:14]:<14} ({row['form / era / buff'][:12]})"
            else:
                cell = ""
            parts.append(f"{cell:<{col_w}}")
        print("|" + "|".join(parts) + "|")
    print("|" + "-" * col_w + "|" + "-" * col_w + "|" + "-" * col_w + "|")


def get_character_input(player_label):
    while True:
        try:
            choice = int(input(f"Select {player_label} (number): "))
            if 1 <= choice <= len(characters):
                return characters.iloc[choice - 1]
            print(f"  Enter 1-{len(characters)}")
        except ValueError:
            print("  Enter a valid number")


def create_matchup_row(char_a, char_b):
    row = {}
    row["a_name"] = char_a["name"]
    row["a_form"] = char_a["form / era / buff"]
    row["a_anime"] = char_a["anime"]
    for col in STAT_COLS:
        row[f"a_{col}"] = char_a[col]

    row["b_name"] = char_b["name"]
    row["b_form"] = char_b["form / era / buff"]
    row["b_anime"] = char_b["anime"]
    for col in STAT_COLS:
        row[f"b_{col}"] = char_b[col]

    return pd.DataFrame([row])


def print_stats_side_by_side(char_a, char_b):
    name_a = f"{char_a['name']} ({char_a['form / era / buff']})"
    name_b = f"{char_b['name']} ({char_b['form / era / buff']})"
    print(f"\n{'STAT':<25} {'A: ' + name_a:<35} {'B: ' + name_b}")
    print("-" * 95)
    for col in STAT_COLS:
        a_val = char_a[col]
        b_val = char_b[col]
        marker = ""
        if a_val > b_val:
            marker = " <<<"
        elif b_val > a_val:
            marker = "                                     <<<"
        print(f"{col:<25} {str(a_val):<35} {str(b_val)}{marker}")


@app.get("/matchups")
def run_matchup(a: int, b: int):
    """Pure logic: takes two character row-IDs, returns the prediction + SHAP as data.
    Used by both the web endpoint and the CLI below."""
    char_a = characters.iloc[a]
    char_b = characters.iloc[b]

    matchup = create_matchup_row(char_a, char_b)
    X = matchup.drop(columns=DROP_COLS)

    proba = model.predict_proba(X)[0]
    winner = char_a["name"] if proba[1] >= 0.5 else char_b["name"]
    confidence = float(max(proba) * 100)

    sv = explainer.shap_values(X)
    pairs = sorted(zip(X.columns.tolist(), sv[0]), key=lambda p: abs(p[1]), reverse=True)
    top_factors = [
        {
            "feature": name,
            "value": round(float(val), 3),               # float() so numpy types serialize to JSON
            "favors": char_a["name"] if val > 0 else char_b["name"],
        }
        for name, val in pairs[:5]
    ]

    return {
        "a_name": char_a["name"], "a_form": char_a["form / era / buff"], "a_anime": char_a["anime"],
        "b_name": char_b["name"], "b_form": char_b["form / era / buff"], "b_anime": char_b["anime"],
        "winner": winner,
        "confidence": round(confidence, 1),
        "proba_a": round(float(proba[1] * 100), 1),
        "proba_b": round(float(proba[0] * 100), 1),
        "top_factors": top_factors,
    }


def build_explain_prompt(char_a, char_b, result):
    """Assemble the narration prompt: canon notes + SHAP factors with the raw A-vs-B
    stat values injected so the LLM can see ties instead of only feature importance."""
    shap_lines = []
    for f in result["top_factors"]:
        stat = f["feature"][2:] if f["feature"][:2] in ("a_", "b_") else f["feature"]
        extra = ""
        if stat in STAT_COLS:
            av, bv = char_a[stat], char_b[stat]
            tie = " [TIED]" if av == bv else ""
            extra = f" | actual {stat}: {char_a['name']}={av} vs {char_b['name']}={bv}{tie}"
        shap_lines.append(f"  {f['feature']}: {f['value']:+.3f} (favors {f['favors']}){extra}")

    return EXPLAIN_PROMPT.format(
        a_name=char_a["name"], a_form=char_a["form / era / buff"], a_anime=char_a["anime"],
        b_name=char_b["name"], b_form=char_b["form / era / buff"], b_anime=char_b["anime"],
        a_notes=char_a["notes"], b_notes=char_b["notes"],
        winner=result["winner"], confidence=result["confidence"],
        shap_factors="\n".join(shap_lines)
    )


def narrate_stream(a: int, b: int):
    """Generator yielding the fight narration token-by-token.
    Shared by the CLI and the /narrate endpoint."""
    char_a = characters.iloc[a]
    char_b = characters.iloc[b]
    result = run_matchup(a, b)
    prompt = build_explain_prompt(char_a, char_b, result)

    try:
        stream = llm.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_EXPLAIN},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,  # gpt-oss is a reasoning model -- reasoning tokens eat the budget before the answer
            temperature=0.3,  # low temp keeps it hewing to the brief instead of embellishing
            timeout=600,
            stream=True
        )
        got_content = False
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                got_content = True
                yield token
        if not got_content:
            yield "[model returned empty content]"
    except Exception as e:
        yield f"[LLM call failed: {type(e).__name__}: {e}]"


@app.get("/narrate")
def narrate(a: int, b: int):
    """Streams the fight narration to the browser token-by-token as it's generated."""
    return StreamingResponse(narrate_stream(a, b), media_type="text/plain")


def predict_matchup():
    print_character_list()

    char_a = get_character_input("Player A")
    char_b = get_character_input("Player B")

    print_stats_side_by_side(char_a, char_b)

    result = run_matchup(char_a.name, char_b.name)   # .name is the row-ID of the selected character

    print(f"\n{'=' * 60}")
    print(f"  WINNER: {result['winner']} ({result['confidence']:.1f}% confidence)")
    print(f"  {result['a_name']}: {result['proba_a']:.1f}%  |  {result['b_name']}: {result['proba_b']:.1f}%")
    print(f"{'=' * 60}")

    print(f"\nTop factors:")
    for f in result["top_factors"]:
        print(f"  {f['feature']:<30} {f['value']:>+.3f}  (favors {f['favors']})")

    print("\nAnalysis:\n  ", end="", flush=True)
    for token in narrate_stream(char_a.name, char_b.name):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    while True:
        predict_matchup()
        again = input("\nSimulate another? (y/n): ").strip().lower()
        if again != "y":
            break
