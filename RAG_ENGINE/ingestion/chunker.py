"""Split knowledge-base markdown files into one chunk per `## ` question.

Expected file shape:

    # Doc Title
    ## Question one?
    Answer...
    ## Question two?
    Answer...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

H1 = re.compile(r"^# (.+)$")
H2 = re.compile(r"^## (.+)$")


@dataclass(frozen=True)
class Chunk:
    source_file: str
    doc_title: str
    question: str
    answer: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """The text that gets embedded and later handed to the LLM.

        The title and question are included so short answers ("E-transfer.")
        still carry enough meaning to be found by a search.
        """
        return f"{self.doc_title} > {self.question}\n\n{self.answer}"


def file_metadata(path: Path) -> dict[str, str]:
    """Derive filterable metadata from names like `07-retailer-walmart-canada.md`."""
    slug = re.sub(r"^\d+-", "", path.stem)
    metadata = {"doc_slug": slug}
    if slug.startswith("retailer-"):
        metadata["retailer"] = slug.removeprefix("retailer-")
    return metadata


def chunk_file(path: Path) -> list[Chunk]:
    lines = path.read_text(encoding="utf-8").splitlines()

    titles = [m.group(1).strip() for line in lines if (m := H1.match(line))]
    if len(titles) != 1:
        raise ValueError(f"{path.name}: expected exactly one '# ' title, found {len(titles)}")
    doc_title = titles[0]

    # (question, body lines). Text above the first `## ` becomes an intro chunk
    # keyed by the doc title, so nothing in the file is silently dropped.
    sections: list[tuple[str, list[str]]] = [(doc_title, [])]
    for line in lines:
        if H1.match(line):
            continue
        if m := H2.match(line):
            sections.append((m.group(1).strip(), []))
        else:
            sections[-1][1].append(line)

    metadata = file_metadata(path)
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for question, body in sections:
        answer = "\n".join(body).strip()
        if not answer:
            if question == doc_title:
                continue  # no intro text, which is the normal case
            raise ValueError(f"{path.name}: '## {question}' has no answer text")
        if question in seen:
            raise ValueError(f"{path.name}: duplicate question '## {question}'")
        seen.add(question)
        chunks.append(Chunk(path.name, doc_title, question, answer, metadata))
    return chunks
