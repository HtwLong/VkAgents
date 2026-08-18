from functools import lru_cache
from pathlib import Path


_SKILL_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load_cv_skill(name: str) -> str:
    """Load a trusted planning playbook by its stable identifier."""
    if not name.replace("-", "").isalnum():
        raise ValueError(f"Invalid CV skill name: {name!r}")
    path = _SKILL_DIR / f"{name}.md"
    if not path.is_file():
        raise ValueError(f"Unknown CV skill: {name}")
    return path.read_text(encoding="utf-8")
