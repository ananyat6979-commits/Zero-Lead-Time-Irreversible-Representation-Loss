# DATA INGESTION — PHASE 2.5
# Single-corpus, public-domain text ingestion.
# No normalization beyond minimal hygiene.

from pathlib import Path
print("RUNNING:", Path(__file__).resolve())
from pathlib import Path


GUTENBERG_HEADER = "*** START OF THE PROJECT GUTENBERG EBOOK"
GUTENBERG_FOOTER = "*** END OF THE PROJECT GUTENBERG EBOOK"


def load_raw_text(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()

    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if "START OF THE PROJECT GUTENBERG EBOOK" in line:
            start_idx = i
        if "END OF THE PROJECT GUTENBERG EBOOK" in line:
            end_idx = i
            break

    if start_idx is None or end_idx is None or end_idx <= start_idx:
        raise ValueError(
            f"Gutenberg boundaries not found or invalid: "
            f"start={start_idx}, end={end_idx}"
        )

    # Skip the header line itself
    content_lines = lines[start_idx + 1 : end_idx]
    return "\n".join(content_lines)


def tokenize(text: str):
    # INTENTIONALLY NAIVE TOKENIZATION
    # No lowercasing, no punctuation stripping.
    # We want natural distributional artifacts.
    return text.split()


def main():
    raw_path = Path("data/raw/pride_and_prejudice.txt")
    out_path = Path("data/processed/pride_and_prejudice.tokens.txt")

    text = load_raw_text(raw_path)
    tokens = tokenize(text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(tokens))

    print(f"Saved {len(tokens)} tokens to {out_path}")
    print("Token count:", len(tokens))
    print("First 20 tokens:", tokens[:20])

    # ---- LINEAGE RECORDING (PHASE 2.5 ONLY) ----
    from src.data.lineage import record_dataset

    record_dataset(
        dataset_path=out_path,
        source="Project Gutenberg: Pride and Prejudice",
        params={
            "tokenization": "whitespace",
            "normalization": "none",
        },
        output_dir=Path("data/lineage"),
    )

if __name__ == "__main__":
    main()
