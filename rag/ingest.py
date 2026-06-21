"""Load and chunk the HR policy markdown into citeable sections.

Chunking is by markdown section: the document's H1 (``# ``) becomes the chunk
``source`` (document title), and each H2 (``## ``) starts a new ``section``. The
intro text between the H1 and the first H2 is kept as an "Overview" chunk. This
keeps citations clean — every chunk maps to a real document + heading.

Parsing has no heavy dependencies, so it is fast and unit-testable on its own;
embeddings are built separately in :mod:`rag.search`.
"""
from __future__ import annotations

import glob
import os

import config


def load_chunks(policies_dir: str | None = None) -> list[dict[str, str]]:
    """Return policy chunks as ``[{id, source, section, text}, ...]``.

    ``id`` is ``"<file-stem>-<section-index>"`` (unique and stable); ``source`` is
    the document title; ``section`` is the heading; ``text`` is heading + body
    (the heading is included so it contributes to the embedding).
    """
    policies_dir = policies_dir or config.POLICIES_DIR
    chunks: list[dict[str, str]] = []

    for path in sorted(glob.glob(os.path.join(policies_dir, "*.md"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()

        title = stem.replace("-", " ").title()
        for line in lines:
            if line.startswith("# ") and not line.startswith("## "):
                title = line[2:].strip()
                break

        sections: list[tuple[str, list[str]]] = []
        heading = "Overview"
        body: list[str] = []

        for line in lines:
            if line.startswith("# ") and not line.startswith("## "):
                continue  # skip the H1 title line itself
            if line.startswith("## "):
                if any(s.strip() for s in body):
                    sections.append((heading, body))
                heading = line[3:].strip()
                body = []
            else:
                body.append(line)
        if any(s.strip() for s in body):
            sections.append((heading, body))

        for index, (section_heading, section_body) in enumerate(sections):
            body_text = "\n".join(section_body).strip()
            if not body_text:
                continue
            chunks.append({
                "id": f"{stem}-{index}",
                "source": title,
                "section": section_heading,
                "text": f"{section_heading}\n{body_text}",
            })

    return chunks
