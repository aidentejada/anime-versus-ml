# Power Scaling Numeric Reference
## Anime Matchup Engine — XGBoost Classifier

---

## Encoding Strategy: Log10(Joules) with Range-Aware Modifiers

Each tier maps to a **log10(Joules)** value. The base tier uses the **midpoint** of its energy range. Modifiers shift toward the low or high end of that range — they are not arbitrary deltas, they are physically grounded positions within the tier's actual energy band.

### Why log10 midpoint as the base

A uniform integer scale lies to the model — it treats every tier gap as equally important when the actual energy differences range from 4x (adjacent sub-tiers) to 25 trillion x (Large Star → Solar System). Log10 encodes magnitude correctly so XGBoost split thresholds are physically meaningful and generalize to unseen matchups.

The midpoint is the right default because a character rated "City level" could be anywhere in that range. The midpoint is the best single estimate. Modifiers then move that estimate toward the low or high end when the wiki gives you that information.

---

## Modifier Rules

| Modifier | Maps To | Rationale |
|---|---|---|
| Base tier (no modifier) | `log10(√(low × high))` — **midpoint** | Best single estimate within the range |
| `Low ___` prefix | `log10(low_end)` | Bottom of the tier's range |
| `High ___` prefix | `log10(high_end)` | Top of the tier's range |
| `+` suffix | `log10(high_end)` | Ceiling of the tier — same as High |
| `likely far higher` | `log10(high_end) + 0.5` | Half log unit above ceiling (~3x nudge above top) |
| `possibly` | midpoint + `(high − mid) × 0.5` | Halfway between mid and high |
| `at least` | midpoint | Floor guarantee, no upward shift |
| `Infinite / Immeasurable / Inapplicable` | **100.0 sentinel** | Capped above all finite tiers |
| `Unknown` / empty | `None` → median impute + binary flag | Never guess |

> **Named `+` tiers in Speed** (`FTL+`, `Supersonic+`, etc.) are their own explicit tiers — the `+` is part of the name, not a modifier. Map them directly, do not apply the `+` modifier logic.

> **Named `+` tiers in AP** (`Universe level+`, `Multiverse level+`, `Outerverse level+`) are also explicit named tiers. These must be matched before the `+` stripping logic fires in the parser.

> **Modifier cap:** No modifier combination should push a value above the next tier's midpoint. Cap at `next_tier_midpoint − 0.01`.

---

## 1. Attack Potency / Striking Strength / Durability / Tier

All four columns share this table. **Base = midpoint. Low prefix = low end. High prefix / `+` = high end.**

| Tier String | Code | Low (log10) | **Mid (log10)** | High (log10) |
|---|---|---|---|---|
| Below Average Human | 10-C | 0.00 | **0.89** | 1.78 |
| Human level | 10-B | 1.78 | **1.90** | 2.03 |
| Athlete level | 10-A | 2.03 | **2.25** | 2.48 |
| Street level | 9-C | 2.48 | **3.33** | 4.18 |
| Wall level | 9-B | 4.18 | **5.75** | 7.32 |
| Small Building level | 9-A | 7.32 | **8.17** | 9.02 |
| Building level | 8-C | 9.02 | **9.47** | 9.92 |
| Large Building level | High 8-C | 9.92 | **10.29** | 10.66 |
| City Block level | 8-B | 10.66 | **11.14** | 11.62 |
| Multi-City Block level | 8-A | 11.62 | **12.12** | 12.62 |
| Small Town level | Low 7-C | 12.62 | **13.00** | 13.39 |
| Town level | 7-C | 13.39 | **14.00** | 14.62 |
| Large Town level | High 7-C | 14.62 | **15.12** | 15.62 |
| Small City level | Low 7-B | 15.62 | **16.02** | 16.42 |
| City level | 7-B | 16.42 | **17.02** | 17.62 |
| Mountain level | 7-A | 17.62 | **18.12** | 18.62 |
| Large Mountain level | High 7-A | 18.62 | **18.94** | 19.26 |
| Island level | 6-C | 19.26 | **19.94** | 20.62 |
| Large Island level | High 6-C | 20.62 | **21.12** | 21.62 |
| Small Country level | Low 6-B | 21.62 | **22.04** | 22.47 |
| Country level | 6-B | 22.47 | **23.04** | 23.62 |
| Large Country level | High 6-B | 23.62 | **24.06** | 24.50 |
| Continent level | 6-A | 24.50 | **24.89** | 25.27 |
| Multi-Continent level | High 6-A | 25.27 | **27.18** | 29.09 |
| Moon level | 5-C | 29.09 | **29.68** | 30.26 |
| Small Planet level | Low 5-B | 30.26 | **31.33** | 32.40 |
| Planet level | 5-B | 32.40 | **33.30** | 34.20 |
| Large Planet level | 5-A | 34.20 | **36.02** | 37.84 |
| Brown Dwarf level | High 5-A | 37.84 | **39.17** | 40.50 |
| Small Star level | Low 4-C | 40.50 | **41.13** | 41.76 |
| Star level | 4-C | 41.76 | **42.13** | 42.50 |
| Large Star level | High 4-C | 42.50 | **43.98** | 45.47 |
| Solar System level | 4-B | 45.47 | **51.38** | 57.30 |
| Multi-Solar System level | 4-A | 57.30 | **61.66** | 66.02 |
| Galaxy level | 3-C | 66.02 | **67.48** | 68.93 |
| Multi-Galaxy level | 3-B | 68.93 | **80.69** | 92.45 |
| Universe level | 3-A | 92.45 | **93.95** | 95.45 |
| High Universe level | — | — | **94.50** | — |
| Universe level+ | — | — | **95.45** | — |
| Low Multiverse level | Low 2-C | — | **96.00** | — |
| Multiverse level | 2-C | — | **96.50** | — |
| Multiverse level+ | 2-B | — | **97.00** | — |
| Low Complex Multiverse level | Low 2-A | — | **97.50** | — |
| Complex Multiverse level | 2-A | — | **98.00** | — |
| High Complex Multiverse level | High 2-A | — | **98.25** | — |
| Hyperverse level | 1-C | — | **98.50** | — |
| High Hyperverse level | High 1-C | — | **98.75** | — |
| Low Outerverse level | 1-B | — | **99.00** | — |
| Outerverse level | 1-A | — | **99.25** | — |
| Outerverse level+ | High 1-A | — | **99.50** | — |
| High Outerverse level | — | — | **99.75** | — |
| **Infinite / Immeasurable / Inapplicable** | — | — | **100.00** | — |

> Tiers above Universe level have no Joule equivalent. Values from High Universe onward are manually spaced to preserve strict ordinality. The model will rarely encounter these in your current dataset.

---

## 2. Speed Scale

Speed uses log10(m/s) of the tier midpoint. **Named `+` tiers are their own explicit entries — do not treat the `+` as a modifier.**

| Tier String | Low (m/s) | High (m/s) | **Value (log10 mid m/s)** |
|---|---|---|---|
| Immobile | 0 | 0 | **0.00** |
| Below Average Human | 0 | 5 | **0.35** |
| Average Human | 5 | 7.7 | **0.79** |
| Athletic Human | 7.7 | 10 | **0.89** |
| Peak Human | 10 | 12.4 | **1.04** |
| Superhuman | 12.4 | 34.3 | **1.33** |
| Subsonic | 34.3 | 171.5 | **1.87** |
| Subsonic+ | 171.5 | 308.7 | **2.37** |
| Transonic | 308.7 | 377.3 | **2.55** |
| Supersonic | 377.3 | 857.5 | **2.76** |
| Supersonic+ | 857.5 | 1715 | **3.07** |
| Hypersonic | 1715 | 3430 | **3.37** |
| Hypersonic+ | 3430 | 8575 | **3.67** |
| High Hypersonic | 8575 | 17150 | **4.07** |
| High Hypersonic+ | 17150 | 34300 | **4.37** |
| Massively Hypersonic | 34300 | 343000 | **4.87** |
| Massively Hypersonic+ | 343000 | 2,997,925 | **5.51** |
| Sub-Relativistic | 2,997,925 | 14,989,623 | **7.33** |
| Sub-Relativistic+ | 14,989,623 | 29,979,246 | **7.52** |
| Relativistic | 29,979,246 | 149,896,229 | **7.85** |
| Relativistic+ | 149,896,229 | 299,792,458 | **8.33** |
| Speed of Light | 299,792,458 | 299,792,458 | **8.48** |
| FTL | 299,792,458 | 2,997,924,580 | **8.93** |
| FTL+ | 2,997,924,580 | 29,979,245,800 | **9.93** |
| Massively FTL | 29,979,245,800 | 299,792,458,000 | **10.93** |
| Massively FTL+ | 299,792,458,000 | open | **11.48** |
| Infinite / Immeasurable / Omnipresent | — | — | **100.00 sentinel** |

> `likely far higher` on any speed tier adds **+0.5** to the tier's value.

---

## 3. Lifting Strength Scale

Log10(kg) midpoint of each class range.

| Tier String | Low (kg) | High (kg) | **Value (log10 mid kg)** |
|---|---|---|---|
| Below Average Human | 0 | 50 | **0.85** |
| Average Human | 50 | 80 | **1.80** |
| Above Average Human | 80 | 120 | **1.99** |
| Athletic Human | 120 | 227 | **2.22** |
| Peak Human | 227 | 545 | **2.55** |
| Superhuman | 545 | 1000 | **2.87** |
| Class 1 | 545 | 1,000 | **2.87** |
| Class 5 | 1,000 | 5,000 | **3.35** |
| Class 10 | 5,000 | 10,000 | **3.85** |
| Class 25 | 10,000 | 25,000 | **4.20** |
| Class 50 | 25,000 | 50,000 | **4.55** |
| Class 100 | 50,000 | 100,000 | **4.85** |
| Class K | 100,000 | 1,000,000 | **5.35** |
| Class M | 10^6 | 10^9 | **7.50** |
| Class G | 10^9 | 10^12 | **10.50** |
| Class T | 10^12 | 10^15 | **13.50** |
| Class P | 10^15 | 10^18 | **16.50** |
| Class E | 10^18 | 10^21 | **19.50** |
| Class Z | 10^21 | 10^24 | **22.50** |
| Class Y | 10^24 | 10^27 | **25.50** |
| Pre-Stellar | 10^27 | 2×10^29 | **28.15** |
| Stellar | 2×10^29 | 3.977×10^32 | **31.05** |
| Multi-Stellar | 3.977×10^32 | 1.6×10^42 | **37.30** |
| Galactic | 1.6×10^42 | 6×10^43 | **42.99** |
| Multi-Galactic | 6×10^43 | 1.5×10^53 | **48.48** |
| Universal | 1.5×10^53 | — | **53.18** |
| Infinite / Immeasurable / Inapplicable | — | — | **100.00 sentinel** |

> `likely far higher` adds **+0.5**. `+` suffix maps to log10(high_end) for that class.

---

## 4. Intelligence Scale

No Joule equivalent exists. Uniform ordinal spacing is correct — these are discrete qualitative levels defined by the wiki, not a physical quantity.

| Tier String | **Value** |
|---|---|
| Mindless | 1 |
| Instinctive | 2 |
| Animalistic | 3 |
| High Animalistic | 4 |
| Below Average | 5 |
| Average | 6 |
| Above Average | 7 |
| Gifted | 8 |
| Genius | 9 |
| Extraordinary Genius | 10 |
| Supergenius | 11 |
| Nigh-Omniscient | 12 |
| Omniscient | 100 sentinel |

> For compound entries ("Below Average normally, Genius in combat") take the **higher** value.

---

## 5. Stamina Scale

| Tier String | **Value** |
|---|---|
| Below Average | 1 |
| Average | 2 |
| Athletic | 3 |
| Peak Human | 4 |
| Superhuman | 5 |
| Infinite / Inapplicable | 100 sentinel |

---

## 6. Worked Examples

### `"Large Town level"` (base, no modifier)
- Mid = **15.12**

### `"Large Town level, likely far higher"`
- High end = 15.62, + 0.5 = **16.12**

### `"Large Town level+"`
- `+` → high end = **15.62**

### `"Small Planet level+"`
- `+` → high end = **32.40**

### `"Planet level, likely far higher"`
- High end (34.20) + 0.5 = **34.70**

### `"FTL+"` (Speed)
- Named tier, direct lookup = **9.93**

### `"Massively FTL+, likely far higher"` (Speed)
- Base value (11.48) + 0.5 = **11.98**

### Naruto 4-B vs Sasuke 5-B
- Naruto AP: Solar System level mid = **51.38**
- Sasuke Dur: Planet level mid = **33.30**
- Delta = **18.08** → ~10^18 x difference → stomp, model learns this fast

---

## 7. Unknown / Missing Value Strategy

| Situation | Encoding |
|---|---|
| `Unknown` / empty | `None` → median impute at train time + binary `{col}_unknown` flag |
| `Varies` | Take the higher listed value + binary `{col}_varies` flag |
| `Inapplicable` | 100.0 sentinel |

---

## 8. Feature Matrix Summary

| Feature | Encoding | Approx range (your dataset) |
|---|---|---|
| `ap_numeric` | log10(J) midpoint | 9.47 – 57.30 |
| `speed_numeric` | log10(m/s) midpoint | 2.76 – 11.48 |
| `durability_numeric` | log10(J) midpoint | 9.47 – 57.30 |
| `striking_strength_numeric` | log10(J) midpoint | 9.47 – 57.30 |
| `lifting_strength_numeric` | log10(kg) midpoint | 2.87 – 37.30 |
| `tier_numeric` | log10(J) midpoint | 9.47 – 57.30 |
| `intelligence_numeric` | ordinal 1–12 | 6 – 10 |
| `stamina_numeric` | ordinal 1–5 (+ 100 sentinel) | 3 – 5 |
| `hax_score` | Groq 1–10 | TBD |
| `regen_score` | Groq 1–10 | TBD |
| `ap_multiplier_score` | Groq 1–10 | TBD |
| `mobility_score` | Groq 1–10 | TBD |
| `resistance_score` | Groq 1–10 | TBD |