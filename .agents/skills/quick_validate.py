#!/usr/bin/env python3
"""Validate the repository's skill entrypoints without external dependencies."""

from __future__ import annotations

import re
import sys
from pathlib import Path


NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
FRONTMATTER_PATTERN = re.compile(r"\A---\n(?P<header>.*?)\n---(?:\n|\Z)", re.DOTALL)


def validate(skill_dir: Path) -> list[str]:
    errors: list[str] = []
    entrypoint = skill_dir / "SKILL.md"
    if not entrypoint.is_file():
        return [f"{skill_dir}: missing SKILL.md"]

    content = entrypoint.read_text(encoding="utf-8")
    match = FRONTMATTER_PATTERN.match(content)
    if match is None:
        return [f"{entrypoint}: invalid or missing YAML frontmatter"]

    fields: dict[str, str] = {}
    for line in match.group("header").splitlines():
        key, separator, value = line.partition(":")
        if not separator or not key.strip() or not value.strip():
            errors.append(f"{entrypoint}: unsupported frontmatter line: {line!r}")
            continue
        fields[key.strip()] = value.strip()

    name = fields.get("name", "")
    description = fields.get("description", "")
    if not NAME_PATTERN.fullmatch(name) or len(name) > 64:
        errors.append(f"{entrypoint}: invalid skill name {name!r}")
    if name != skill_dir.name:
        errors.append(
            f"{entrypoint}: name {name!r} does not match directory {skill_dir.name!r}"
        )
    if not description or len(description) > 1024:
        errors.append(f"{entrypoint}: description must contain 1-1024 characters")
    if "[TODO:" in content:
        errors.append(f"{entrypoint}: unfinished TODO placeholder")

    return errors


def main(arguments: list[str]) -> int:
    root = Path(__file__).resolve().parent
    targets = [Path(argument) for argument in arguments]
    if not targets:
        targets = sorted(path for path in root.iterdir() if path.is_dir())

    errors = [error for target in targets for error in validate(target)]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    print(f"Validated {len(targets)} skill director{'y' if len(targets) == 1 else 'ies'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
