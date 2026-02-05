# scripts/expand_structured.py
import json
import random
import argparse
from pathlib import Path

AGES = [6, 7, 8, 9, 10, 11, 12, 13, 14]

THEMES = [
    ("sharing", "Sharing makes play kinder for everyone."),
    ("honesty", "Telling the truth is brave and helps people trust each other."),
    ("kindness", "Small kindness can change someone’s whole day."),
    ("patience", "Good things often take time and practice."),
    ("perseverance", "Trying again is how you get better."),
    ("friendship", "Friends listen, support, and forgive."),
    ("empathy", "Understanding others helps you choose kinder actions."),
    ("responsibility", "Taking responsibility means fixing mistakes, not hiding them."),
    ("teamwork", "Working together makes hard things easier."),
    ("bullying", "It’s strong to ask for help and stand up calmly for respect."),
    ("self-confidence", "You can be nervous and still be capable."),
    ("study-habits", "Small routines build big progress."),
    ("internet-safety", "Being safe online means being careful with what you share."),
    ("respect", "Respect shows in words, tone, and choices."),
    ("gratitude", "Noticing good things makes you feel lighter."),
]

SETTINGS = [
    "at home", "at school", "in the library", "in a small park", "at a community center",
    "on a rainy afternoon", "during a class project", "before bedtime", "on a weekend morning",
]

PROTAGONISTS = [
    ("Lina", "she"), ("Omar", "he"), ("Mina", "she"), ("Youssef", "he"),
    ("Noah", "he"), ("Sara", "she"), ("Adam", "he"), ("Lea", "she"),
    ("Nina", "she"), ("Ilyes", "he"), ("Aya", "she"), ("Rayan", "he"),
]

HELPERS = [
    "a parent", "a teacher", "a friend", "an older sibling", "the school counselor",
    "a librarian", "a coach", "a neighbor",
]

FUNNY_PROBLEMS = [
    "a backpack zipper gets stuck at the worst time",
    "a poster board bends and flops like a sleepy pancake",
    "a marker leaks and makes a surprise dot on a sleeve",
    "a phone autocorrects a message into something silly",
    "a homework page gets shuffled with the wrong class",
]

def make_outline(theme: str) -> list[str]:
    # 5-beat outline, consistent structure
    return [
        "Introduce the main character and what they want",
        f"Problem: a situation tests the idea of {theme}",
        "A small consequence happens and feelings appear",
        "A helper offers a simple, practical idea",
        "The character acts differently and the story ends happily with a lesson",
    ]

def make_story(name: str, pronoun: str, age: int, theme: str, lesson: str, setting: str) -> str:
    helper = random.choice(HELPERS)
    funny = random.choice(FUNNY_PROBLEMS)

    # Keep it kid/teen friendly, 180–300ish words, clear ending + explicit lesson line.
    if age <= 9:
        voice = "simple"
    else:
        voice = "slightly_older"

    if theme == "internet-safety":
        situation = (
            f"{name} wanted to post a photo {setting}, because it looked cool. "
            f"But the photo also showed a school sign in the background."
        )
        fix = (
            f"{name} asked {helper} what to do. The {helper} said, "
            f"“Share fun moments, but hide details that tell strangers where you are.” "
            f"{name} cropped the photo and used a friendly caption instead."
        )
    elif theme == "bullying":
        situation = (
            f"{name} heard someone tease a classmate {setting}. "
            f"{name} felt a tight knot in the stomach and didn’t know what to say."
        )
        fix = (
            f"{name} told {helper} what happened. The {helper} suggested three calm steps: "
            f"check on the classmate, name the behavior as not okay, and ask an adult for support. "
            f"{name} did exactly that."
        )
    else:
        situation = (
            f"{name} {setting} really wanted things to go perfectly, but {funny}. "
            f"{name} felt frustrated and started to rush."
        )
        fix = (
            f"{name} talked to {helper}. The {helper} suggested a small plan: "
            f"pause, take one slow breath, choose one helpful action, and continue."
        )

    # Theme-specific wrap
    if theme == "sharing":
        twist = f"{name} noticed a friend watching quietly, wanting a turn."
        action = f"{name} offered a turn and suggested playing together."
    elif theme == "honesty":
        twist = f"{name} almost blamed the mistake on “someone else,” but it didn’t feel right."
        action = f"{name} told the truth and helped fix the problem."
    elif theme == "patience":
        twist = f"{name} wanted the result immediately, but it didn’t work that way."
        action = f"{name} tried again slowly and waited for progress."
    elif theme == "perseverance":
        twist = f"{name} failed the first try and wanted to quit."
        action = f"{name} practiced one small part, then tried again."
    elif theme == "study-habits":
        twist = f"{name} felt overwhelmed by a big task."
        action = f"{name} broke it into 10-minute steps and checked them off."
    elif theme == "gratitude":
        twist = f"{name} had a rough day and only noticed what went wrong."
        action = f"{name} wrote down three small good things and felt calmer."
    elif theme == "internet-safety":
        twist = f"{name} realized “cool” isn’t worth sharing private details."
        action = f"{name} adjusted the post and felt proud of being careful."
    elif theme == "bullying":
        twist = f"{name} chose courage without yelling or fighting."
        action = f"{name} supported the classmate and involved a trusted adult."
    else:
        twist = f"{name} noticed someone else was affected by the situation too."
        action = f"{name} chose a kinder, more thoughtful response."

    # Slightly older voice for 10–14 (more reflective, still safe + clear)
    if voice == "slightly_older":
        ending = (
            f"By the end of the day, {name} felt more in control—not because everything was perfect, "
            f"but because {pronoun} handled it with care. {lesson}"
        )
    else:
        ending = (
            f"That night, {name} smiled and felt proud. {lesson}"
        )

    story = (
        f"{name} was {age} years old. One day {setting}, {situation} "
        f"{twist} {fix} "
        f"{action} "
        f"Everyone felt better, and the problem got smaller instead of bigger. "
        f"{ending}"
    )

    return story.strip()

def build_examples(n: int, seed: int = 42) -> list[dict]:
    random.seed(seed)
    examples = []
    used = set()

    while len(examples) < n:
        age = random.choice(AGES)
        name, pronoun = random.choice(PROTAGONISTS)
        theme, lesson = random.choice(THEMES)
        setting = random.choice(SETTINGS)

        key = (age, name, theme, setting)
        if key in used:
            continue
        used.add(key)

        ex = {
            "instruction": f"Write a story for a {age}-year-old about {theme.replace('-', ' ')}. Keep it positive, clear, and end with a lesson.",
            "outline": make_outline(theme.replace('-', ' ')),
            "story": make_story(name, pronoun, age, theme, lesson, setting),
        }
        examples.append(ex)

    return examples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="Number of stories to generate")
    ap.add_argument("--out", type=str, default="data/structured.jsonl", help="Output JSONL path")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = build_examples(args.n, args.seed)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(examples)} examples to {out_path}")

if __name__ == "__main__":
    main()