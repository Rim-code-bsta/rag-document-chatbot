# feedback_stats.py  petit script d'analyse des logs de feedback
# Usage : lancez-le depuis le même dossier que streamlit_app.py, une fois que
# vous avez utilisé le chatbot un moment (les fichiers feedback_log.jsonl et
# unanswered_questions.jsonl se remplissent automatiquement au fil de l'usage).
#
#   python feedback_stats.py
#
# Utile pour sortir des chiffres concrets à mettre dans le rapport, par ex :
# "X% des réponses évaluées ont reçu un retour négatif", ou la liste des
# questions pour lesquelles le chatbot n'a pas trouvé d'information.

import json
from collections import Counter
from pathlib import Path

FEEDBACK_LOG   = Path("feedback_log.jsonl")
UNANSWERED_LOG = Path("unanswered_questions.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def print_feedback_stats():
    entries = load_jsonl(FEEDBACK_LOG)
    print("=" * 60)
    print(f"FEEDBACK — {len(entries)} retour(s) enregistré(s)")
    print("=" * 60)

    if not entries:
        print("Aucun feedback enregistré pour le moment.\n")
        return

    counts = Counter(e["feedback"] for e in entries)
    total  = len(entries)
    likes  = counts.get("like", 0)
    dislikes = counts.get("dislike", 0)

    print(f"👍 Utile      : {likes}  ({likes/total*100:.1f}%)")
    print(f"👎 Pas utile  : {dislikes}  ({dislikes/total*100:.1f}%)")
    print()

    if dislikes:
        print("Questions ayant reçu un retour négatif :")
        for e in entries:
            if e["feedback"] == "dislike":
                print(f"  - [{e['timestamp']}] {e['question']}")
        print()


def print_unanswered_stats():
    entries = load_jsonl(UNANSWERED_LOG)
    print("=" * 60)
    print(f"QUESTIONS SANS RÉPONSE — {len(entries)} question(s) enregistrée(s)")
    print("=" * 60)

    if not entries:
        print("Aucune question sans réponse enregistrée pour le moment.\n")
        return

    for e in entries:
        print(f"  - [{e['timestamp']}] {e['question']}")
    print()


if __name__ == "__main__":
    print_feedback_stats()
    print_unanswered_stats()