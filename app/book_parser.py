"""
book_parser.py – Parse the Ayurvedic book JSON into search-optimised chunks.

Each medicinal plant entry is split into multiple small, semantically focused
chunks so the RAG retriever can surface the most relevant treatment quickly.

Chunk strategy (designed for high recall + precision):
  1. **Plant overview chunk** — plant name, botanical name, family, all
     vernacular names.  Ensures name-based queries ("What is Ashwagandha?")
     match even without ailment keywords.
  2. **One chunk per ailment** — contains the plant context *plus* the full
     treatment paragraph.  Keeps each chunk short and topically focused,
     which improves both embedding quality and BM25 keyword hits.
"""
from __future__ import annotations

import json
import os
from typing import Any

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
BOOK_JSON = os.path.join(DATA_DIR, "book.json")


def _vernacular_text(names: dict[str, str]) -> str:
    """Format vernacular names into a readable line."""
    return ", ".join(f"{lang}: {name}" for lang, name in names.items())


def _plant_header(plant: dict[str, Any]) -> str:
    """Build a reusable header string for a plant entry."""
    parts = [
        f"Plant: {plant['plant_name']}",
        f"Botanical name: {plant.get('botanical_name', 'N/A')}",
        f"Family: {plant.get('family', 'N/A')}",
    ]
    return " | ".join(parts)


def parse_book_to_chunks(json_path: str | None = None) -> list[dict]:
    """
    Parse the Ayurvedic book JSON and return a list of chunk dicts.

    Each dict has:
      - id        : deterministic string id  (e.g. ``book::plant_3::overview``)
      - text      : the chunk text (optimised for embedding + BM25)
      - metadata  : dict with source, plant_name, botanical_name, family,
                     chunk_type ("overview" | "treatment"), and optionally
                     ailment.

    Returns
    -------
    list[dict]
    """
    if json_path is None:
        json_path = BOOK_JSON

    with open(json_path, "r", encoding="utf-8") as f:
        plants: list[dict] = json.load(f)

    chunks: list[dict] = []

    for plant in plants:
        pid = plant.get("id", "?")
        pname = plant.get("plant_name", "Unknown")
        botanical = plant.get("botanical_name", "")
        family = plant.get("family", "")
        vernacular = plant.get("vernacular_names", {})

        header = _plant_header(plant)
        vern_text = _vernacular_text(vernacular) if vernacular else ""

        # ── 1. Overview chunk ──────────────────────────────────────────
        overview_lines = [
            header,
            f"Vernacular names — {vern_text}" if vern_text else "",
        ]
        # list ailments this plant treats (helps retrieval)
        ailment_names = [
            u["ailment"]
            for u in plant.get("medicinal_uses", [])
            if "ailment" in u
        ]
        if ailment_names:
            overview_lines.append(
                f"Medicinal uses: treats {', '.join(ailment_names)}."
            )

        overview_text = "\n".join(ln for ln in overview_lines if ln)

        chunks.append(
            {
                "id": f"book::plant_{pid}::overview",
                "text": overview_text,
                "metadata": {
                    "source": "book.json",
                    "plant_name": pname,
                    "botanical_name": botanical,
                    "family": family,
                    "chunk_type": "overview",
                },
            }
        )

        # ── 2. One chunk per ailment/treatment ─────────────────────────
        for idx, use in enumerate(plant.get("medicinal_uses", [])):
            ailment = use.get("ailment", "General")
            treatment = use.get("treatment", "")

            treatment_text = (
                f"{header}\n"
                f"Ailment: {ailment}\n"
                f"Treatment: {treatment}"
            )

            chunks.append(
                {
                    "id": f"book::plant_{pid}::treat_{idx}",
                    "text": treatment_text,
                    "metadata": {
                        "source": "book.json",
                        "plant_name": pname,
                        "botanical_name": botanical,
                        "family": family,
                        "chunk_type": "treatment",
                        "ailment": ailment,
                    },
                }
            )

    return chunks


if __name__ == "__main__":
    chunks = parse_book_to_chunks()
    print(f"Parsed {len(chunks)} chunks from book.json")
    # show first 3 samples
    for c in chunks[:3]:
        print(f"\n--- {c['id']} ---")
        print(c["text"][:300])
