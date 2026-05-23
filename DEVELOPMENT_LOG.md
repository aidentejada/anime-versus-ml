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

#### Entry 1:
I ran into a roadblock looking for data today. I spend hours on kaggle, tried web scraping to no avail, and then i realized, its better that I curate this list of 60-70 characters by hand. That way my data is of good quality, and it is bound to work and fit my needs. Although having a robust data pipeline would look so good on a project like this, the reality of data is that sometimes it has to be collected manually.

Additionally I got to thinking, if i pull these stats from the wiki, and use my own weights to determine who wins what match up (1 or 0) and then run that through a classifier model, what is the model useful for? The answer is, nothing really. The issue with using my own algorithmic/deterministic methods is that anime is so nuanced, different environments, different power types, different feats, and different power scaling. So my solution for that is to feed the model the raw stats along with the winner, and let it decide how to weigh each stat and vet them against eachother. What flourishes in these types of projects is domain knowledge, simply knowing who would win in a matchup and why. However with 70 characters, I have a few thousand matchups to decide the winner for which is not practical to say the very least. What I can do is send API requests to an LLM (groq) to decide and return to me the winner (1 or 0) WITHOUT seeing the stats I have in place. This will allow the model to determine its own formulas and weights and have a more practical use in this project (and hopefully catch those nuances i may have missed in a hardcoded deterministic approach)
