import json
import random
from pathlib import Path

OUTPUT_FILE = Path("data/raw/questcrafter_pairs.jsonl")

SETTINGS = ["forest", "mountains", "desert", "castle", "swamp"]
TONES = ["humorous", "dark", "epic", "mysterious", "wholesome"]

TITLES = [
    "The Trial of the Talking Goat",
    "The Goblin Who Wanted a Hug",
    "The Dragon With Stage Fright",
    "The Quest for the Lost Sandwich",
    "The Mountain That Sneezed"
]

def generate_sample():
    setting = random.choice(SETTINGS)
    tone = random.choice(TONES)
    title = random.choice(TITLES)

    prompt = f"""
You are QuestCrafter, an AI Dungeon Master.

LEVEL: 9
SETTING: {setting}
TONE: {tone}
LENGTH: medium

TASK: Write a fantasy RPG quest with Title, Hook, Objective, Steps, Twist, Reward.
"""

    response = f"""TITLE: {title}

HOOK: Strange trouble spreads through the {setting}, and only a brave adventurer can help.

OBJECTIVE: Investigate the mystery and restore safety.

STEPS:
1. Gather clues from locals.
2. Overcome a guarded obstacle.
3. Face the final confrontation.

TWIST: The ally guiding you is secretly part of the problem.

REWARD: Gold, a rare magical item, and fame worthy of a level-10 hero.
"""

    return {"text": f"<|user|>{prompt}<|assistant|>{response}"}


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    N =30000  

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for _ in range(N):
            sample = generate_sample()
            f.write(json.dumps(sample) + "\n")

    print(f"✅ Generated {N} samples into {OUTPUT_FILE}")


if __name__ == "__main__":
    main()