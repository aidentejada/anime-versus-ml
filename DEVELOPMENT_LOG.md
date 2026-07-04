### Day One: Skeleton Setup (*05/21/2026*)  
#### Summary:  
- Set up skeleton of the project, I want to make this "Production-grade" so I looked up details on how I can achieve that, and it entails making separate folders, and an `__init__.py` file in the src folder which i've never done before.   
- Additionally I added a `requirements.txt` along with a frozen snapshot of all the versions of the packages I used in this project. I did this to ensure my code doesn't break in a few years due to package updates.  
- I also initialized a virtual environment and added it to the .gitignore file.  
  
#### Notes / Thoughts:  
- I have an Idea of what I want to do, but I need to break it down into smaller tasks. Thats why before I start coding I would like to write the `README.md` to refer back to and track milestones.  
  
### Day Two: README.md and Data Pipeline (*05/22/2026*)  
#### Summary:  
- Created a `README.md` file to refer back to and track milestones.  
- Hand curated a list of anime characters and manually pulled their consensus stats from vsbattles.fandom.com (could be automated but would break the 80/20 rule)  
- Using the following features:  
  - Attack Potency (AP)  
  - Speed  
  - Intelligence  
  - Durability  
  - Stamina  
  - Lifting Strength  
  - Striking Strength  
  - Range  
  - Number of Abilities  
  - Number of Resistances  
  - "Hax" Score  
  
#### Entry:  
I ran into a roadblock looking for data today. I spend hours on kaggle, tried web scraping to no avail, and then i realized, its better that I curate this list of 60-70 characters by hand. That way my data is of good quality, and it is bound to work and fit my needs. Although having a robust data pipeline would look so good on a project like this, the reality of data is that sometimes it has to be collected manually.  
  
Additionally I got to thinking, if i pull these stats from the wiki, and use my own weights to determine who wins what match up (1 or 0) and then run that through a classifier model, what is the model useful for? The answer is, nothing really. The issue with using my own algorithmic/deterministic methods is that anime is so nuanced, different environments, different power types, different feats, and different power scaling. So my solution for that is to feed the model the raw stats along with the winner, and let it decide how to weigh each stat and vet them against eachother. What flourishes in these types of projects is domain knowledge, simply knowing who would win in a matchup and why. However with 70 characters, I have a few thousand matchups to decide the winner for which is not practical to say the very least. What I can do is send API requests to an LLM (groq) to decide and return to me the winner (1 or 0) WITHOUT seeing the stats I have in place. This will allow the model to determine its own formulas and weights and have a more practical use in this project (and hopefully catch those nuances i may have missed in a hardcoded deterministic approach)  
  
### Day Three: Manual Data Collection + Labeling (*05/23/2026*)  
#### Summary:  
- Manually went through the process of collecting stats for each character. the wiki categorizes them by quantifiable values, e.g. "Mountain Level", "Planetary Level", and their corresponding Joules of output, same thing with speed in m/s. so I'm just taking the log10(middle of joule range) for tier and other modifiers aswell. I had claude generate an entire document called POWER_SCALE.md for further details  
- Created Files:
	- llmrequest.py
	- prompts.py
	- matchup_maker.py
	- characters_stats_text.csv
	- characters_stats_numerical.csv
	- characters_scored_final.csv
	- matchups_to_label.csv
	- matchups_winners.csv
#### Entry:
Today was quite the busy day. I dedicated about 10 hours total to the project, manually collecting data from the wiki. `/characters_stats_text.csv` uses power scaling quantifiers in words like 'Planetary level' or 'Massively Faster Than Light' luckily, the power scaling wiki gives numerical values for this already, so I can just use that to map out `/characters_stats_numerical.csv` which is numerical / quantifiable by a model. I decided to use log10(middle of joule range) for the durability and tier, and upper range for modifiers like + or things like "High 8c tier, likely more" all the modiers add a certain amount to that numerical value, the entire guide is in `POWER_SCALE.md`. Then, since most of the characters within the same universe are similar tier and stregth, what could be used to differentiate them? or how could I justify a Continental level character beating a Planetary level character? This could probably break the model if not justified by some numerical difference and is exactly why I added ability columns:
```json
"durability_negation":  {"type": "number"},  
"regeneration":         {"type": "number"},  
"power_amplification":  {"type": "number"},  
"mobility_hax":         {"type": "number"},  
"time_space_manip":     {"type": "number"},  
"mind_soul_hax":        {"type": "number"},  
"resistance_physical":  {"type": "number"},  
"resistance_hax":       {"type": "number"}
```
This would obviously be a pain to scour the internet or use my own very limited and probably biased domain knowledge to label 60 characters. So i created `llmrequest.py` to use AI as a labeler. For this, I spent the money to ensure my data was actually good, and opted for Anthropic's Claude Sonnet 4.5 and its deep reasoning and domain knowledge to assign those abilities. Now the model can capture those nuances and those weird abilities that come with simuliting multi-universe matchups. After labeling, i created `matchup_maker.py`to go through and make matchups for every combination of all the characters, meaning I'm treating a vs b the same as b vs a, (permutations will be used for training, but to label winners its easier and to make the combinations, label those and then mirror the dataset to double the amount of data, this ensures consistancy and prevents contradictions like character a wins in `a vs b`, but b wins in `b vs a`. that would confuse the model. After running `matchup_maker.py` I got `matchups_to_label.csv` which contained in each row character a, all its stats, and character b with all its stats and an empty winner row, 1 = character a wins, 0 = character a loses, or conversely character b wins. Using `llmrequest.py` again, I wrote a function and another prompt (check `prompts.py`) to label the winners using domain knowledge. Because it was 2000+ rows, the more economic choice was Claude Haiku 4.5 but this would prove to be a mistake. As it was labeling I saw some crazy misjudgment in the console. Discouraged, stopped here.

### Day Four: Fixing Data & Training Model (*06/27/2026*)  
#### Summary:
- Took a long break from the project because of the incorrect labels. Fixed them, Trained the model, tuned hyperparameters.
- Model Achieved a 91.55% accuracy against 5 cross validation rounds.
- SHAP identified the most important factors in deciding a matchup (tier, durability, speed)
- Created Files:
	- matchups_winners_fixed.csv
	- mirror_matchups.py
	- matchups_winners_final.csv
	- train.py
	- predict.py
#### Entry:
I wasn't going to let the time i spent manually collecting high-quality data go to waste. So I broke the 2000+ rows into 24 batches of 100 matchups, and went 100 at a time across weeks of maxing out my usage limits, analyzing each batch to find misjudgements, and mark what the true outcome of those incorrect matchups should be. After which I could just make a short python script to correct the rows in order to train the model properly. I then made `mirror_matchups.py` to double the training data by mirroring the combinations we already had. This left me with 4000+ rows of high quality, carefully curated training data `matchups_winners_final.csv`. I then trained the model with no hyperparameter tuning with an 80 / 20 train test split. Original Accuracy was 90% with the most important factor being speed (which makes sense due to speed blitzing). After using `GridSearchCV` to tune the hyperparameters, I got up to an accuracy of 92%, and the most important factors according to shap was tier, durability, then speed. I saved the model to use for live inference.

### Day Five: Fixing model,Fast API work. (*07/02/2026*)
I came to the realization that the model training had leakage meaning out 91.55% score isn't very accurate. We doubled the training data from 2346 to 4692 rows, but when we train-test-split, the same matchups that are in the training data are also in the test data, which is the root of the leakage. I fixed this and started implementing Fast API for deployment.
Entry: