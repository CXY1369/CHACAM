from pathlib import Path
import re


def test_source_tree_is_species_agnostic():
    source_root = Path(__file__).parents[1] / "src" / "chacam"
    forbidden = re.compile(r"\b(elegans|WormBase|Tintori|Cole|CShaper|ABal)\b")
    hits = []
    for path in sorted(source_root.rglob("*.py")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if forbidden.search(line):
                hits.append(f"{path}:{line_number}: {line.strip()}")
    assert not hits, "forbidden species-specific terms found:\n" + "\n".join(hits)
