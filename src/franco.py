"""
Shared Franco-Arabic lexicon utilities.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_FRANCO_SEED_PATH = DATA_ROOT / "franco_seed.csv"

GENERIC_ENGLISH_FRANCO_ENTRIES = {
    "perfect",
    "amazing",
    "nice",
    "love it",
}

FRANCO_LABEL_BLOCKS: Dict[str, Dict[str, str]] = {
    "positive": {
        "7elw": "حلو",
        "helw": "حلو",
        "7elwa": "حلوة",
        "helwa": "حلوة",
        "gamda": "جامدة",
        "gamed": "جامد",
        "gameel": "جميل",
        "gamila": "جميلة",
        "momtaz": "ممتاز",
        "momtaza": "ممتازة",
        "to7fa": "تحفة",
        "tuhfa": "تحفة",
        "raw3a": "روعة",
        "rawaa": "روعة",
        "3agabny": "عجبني",
        "3agbny": "عجبني",
        "perfect": "ممتاز",
        "amazing": "رائع",
        "nice": "حلو",
        "love it": "عجبني",
        "7abeto": "حبيته",
        "habeto": "حبيته",
        "7abetaha": "حبيتها",
        "kwayes": "كويس",
        "kowayes": "كويس",
        "kwayesa": "كويسة",
        "tamam": "تمام",
        "fol": "فل",
        "fol awy": "فل أوي",
        "zy el fol": "زي الفل",
        "mozbot": "مظبوط",
        "saree3": "سريع",
        "nedeef": "نضيف",
        "nadeef": "نضيف",
    },
    "negative": {
        "wahesh": "وحش",
        "w7esh": "وحش",
        "we7esh": "وحش",
        "zift": "زفت",
        "khara": "سيء",
        "say2": "سيء",
        "saye2": "سيء",
        "msh 7elw": "مش حلو",
        "mesh helw": "مش حلو",
        "msh kwayes": "مش كويس",
        "mesh kwayes": "مش كويس",
        "msh tamam": "مش تمام",
        "ba2eza": "باظت",
        "bazet": "باظت",
        "baye5": "بايخ",
        "bayekh": "بايخ",
        "mo2ref": "مقرف",
        "me2ref": "مقرف",
        "ghaly": "غالي",
        "8aly": "غالي",
        "mot2akher": "متأخر",
        "mota5ar": "متأخر",
        "batee2": "بطيء",
        "bati2": "بطيء",
        "msh nedeef": "مش نضيف",
        "mesh nadeef": "مش نضيف",
        "daye3": "ضايع",
        "modya3": "مضيع",
        "msh 3agabny": "مش عاجبني",
        "mesh 3agbny": "مش عاجبني",
    },
    "neutral": {
        "ah": "اه",
        "aah": "اه",
        "aywa": "ايوه",
        "la2": "لا",
        "laa": "لا",
        "msh": "مش",
        "mesh": "مش",
        "ana": "انا",
        "enta": "انت",
        "enty": "انتي",
        "howa": "هو",
        "heya": "هي",
        "eh": "ايه",
        "leh": "ليه",
        "feen": "فين",
        "emta": "امتى",
        "kam": "كام",
        "da": "ده",
        "de": "دي",
        "dol": "دول",
    },
}

FRANCO_MAP = {
    franco: arabic
    for mapping in FRANCO_LABEL_BLOCKS.values()
    for franco, arabic in mapping.items()
}


def iter_franco_seed_rows() -> Iterable[Dict[str, str]]:
    """Yield Franco seed rows in a stable order."""
    for label, mapping in FRANCO_LABEL_BLOCKS.items():
        for franco, arabic in mapping.items():
            yield {
                "franco": franco,
                "arabic": arabic,
                "label": label,
            }


def get_franco_entries() -> List[Dict[str, str]]:
    """Return the lexicon rows as a list of dictionaries."""
    return list(iter_franco_seed_rows())


def write_franco_seed_csv(path: Path = DEFAULT_FRANCO_SEED_PATH) -> Path:
    """Persist the Franco seed lexicon as UTF-8 CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["franco", "arabic", "label"])
        writer.writeheader()
        writer.writerows(get_franco_entries())
    return path


def load_franco_seed(path: Path = DEFAULT_FRANCO_SEED_PATH) -> List[Dict[str, str]]:
    """Load the Franco seed CSV when present, otherwise return the built-in lexicon."""
    if not path.exists():
        return get_franco_entries()

    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "franco": str(row.get("franco", "")).strip(),
                    "arabic": str(row.get("arabic", "")).strip(),
                    "label": str(row.get("label", "")).strip() or "neutral",
                }
            )
    return rows


def load_franco_map(path: Path = DEFAULT_FRANCO_SEED_PATH) -> Dict[str, str]:
    """Load the Franco-to-Arabic map from disk or the embedded fallback."""
    return {
        row["franco"]: row["arabic"]
        for row in load_franco_seed(path)
        if row.get("franco") and row.get("arabic")
    }


def load_labeled_franco_map(path: Path = DEFAULT_FRANCO_SEED_PATH) -> Dict[str, Dict[str, str]]:
    """Load the full Franco lexicon keyed by source phrase."""
    return {
        row["franco"]: {
            "arabic": row["arabic"],
            "label": row["label"],
        }
        for row in load_franco_seed(path)
        if row.get("franco") and row.get("arabic")
    }


if __name__ == "__main__":
    output_path = write_franco_seed_csv()
    print(f"Wrote {len(get_franco_entries())} Franco seed rows to {output_path}")
