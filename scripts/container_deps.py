#!/usr/bin/env python3
"""Derive container relationships from Dockerfiles.

The base image of every container is declared exactly once, in its Dockerfile.
This module resolves that declaration for multi-stage builds and classifies it,
so that CI does not need a second, drift-prone copy of the dependency graph.

Run as a script it prints a JSON map:

    $ container_deps.py
    {
      "python-apps-base": {
        "base": "${REGISTRY}app-bricks/python-base:${BASE_IMAGE_VERSION}",
        "parent": "python-base"
      },
      ...
    }

``base`` is the external image the final stage builds on, with build-arg
placeholders left intact. ``parent`` is the container name when the base is
another container of this repository, otherwise null.

The ``closure`` subcommand widens a selection with the containers deriving from
it, so CI builds a consistent set; bake orders the builds through its parent
links:

    $ container_deps.py closure python-base models-downloader
    ["models-downloader", "python-apps-base", "python-base"]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

FROM_PATTERN = re.compile(r"^\s*FROM\s+(?:--platform=\S+\s+)?(\S+)(?:\s+AS\s+(\S+))?\s*$", re.IGNORECASE)
PARENT_PATTERN = re.compile(r"^\$\{REGISTRY\}app-bricks/([a-z0-9-]+):\$\{BASE_IMAGE_VERSION\}$")


def resolve_base_image(dockerfile: Path) -> str:
    """Return the external image the Dockerfile's final stage builds on.

    Multi-stage builds produce the last stage, so resolution starts there and
    follows ``FROM <alias>`` references through earlier stages until it reaches
    an image that is not a stage of the same Dockerfile.
    """
    stages: list[tuple[str, str | None]] = []
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        match = FROM_PATTERN.match(line)
        if match:
            stages.append((match.group(1), match.group(2)))
    if not stages:
        raise ValueError(f"No FROM instruction in {dockerfile}")

    aliases = {alias.lower(): base for base, alias in stages if alias}
    base = stages[-1][0]
    seen: set[str] = set()
    while base.lower() in aliases:
        if base.lower() in seen:
            raise ValueError(f"Circular stage references in {dockerfile}")
        seen.add(base.lower())
        base = aliases[base.lower()]
    return base


def parent_container(base_image: str) -> str | None:
    """Return the container name when the base image is built by this repository."""
    match = PARENT_PATTERN.match(base_image)
    return match.group(1) if match else None


def container_map(containers_dir: Path) -> dict[str, dict[str, str | None]]:
    """Map every container to its resolved base image and parent container."""
    result: dict[str, dict[str, str | None]] = {}
    for dockerfile in sorted(containers_dir.glob("*/Dockerfile")):
        base = resolve_base_image(dockerfile)
        result[dockerfile.parent.name] = {"base": base, "parent": parent_container(base)}
    return result


def closure(
    containers_dir: Path,
    selection: list[str],
    expand_parents: bool,
    exclude: list[str] | None = None,
) -> list[str]:
    """Widen a container selection so related images stay consistent.

    The containers deriving from the selection are added, so a parent is never
    rebuilt without its children. With ``expand_parents`` the parents of
    selected containers are added too, so they are published alongside a child
    that was rebuilt on them. ``exclude`` stops the descendant expansion at the
    given containers, e.g. at those released by another tag prefix.
    """
    deps = container_map(containers_dir)
    unknown = sorted(set(selection) - deps.keys())
    if unknown:
        raise ValueError(f"Unknown containers: {', '.join(unknown)}")

    excluded = set(exclude or []) - set(selection)
    selected = set(selection)
    while True:
        descendants = {name for name, info in deps.items() if info["parent"] in selected} - selected - excluded
        if not descendants:
            break
        selected |= descendants

    if expand_parents:
        frontier = selected
        while frontier:
            parents = {parent for name in frontier if (parent := deps[name]["parent"])} - selected
            selected |= parents
            frontier = parents
    return sorted(selected)


def tree(containers_dir: Path) -> str:
    """Render the container hierarchy grouped by external base image."""
    deps = container_map(containers_dir)
    children: dict[str | None, list[str]] = {}
    for name, info in deps.items():
        children.setdefault(info["parent"], []).append(name)

    lines: list[str] = []

    def render(name: str, prefix: str) -> None:
        branch = children.get(name, [])
        for i, child in enumerate(sorted(branch)):
            last = i == len(branch) - 1
            lines.append(f"{prefix}{'└─' if last else '├─'} {child}")
            render(child, prefix + ("   " if last else "│  "))

    roots = sorted(name for name, info in deps.items() if info["parent"] is None)
    for base in sorted({deps[r]["base"] for r in roots}):
        lines.append(base)
        for root in (r for r in roots if deps[r]["base"] == base):
            last = root == max(r for r in roots if deps[r]["base"] == base)
            lines.append(f"{'└─' if last else '├─'} {root}")
            render(root, "   " if last else "│  ")
        lines.append("")
    return "\n".join(lines).rstrip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    closure_parser = subparsers.add_parser("closure", help="Widen a selection with related containers.")
    closure_parser.add_argument("containers", nargs="+", help="Selected container names.")
    closure_parser.add_argument(
        "--expand-parents",
        action="store_true",
        help="Also include the parents of selected containers.",
    )
    closure_parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Containers the descendant expansion must not cascade into.",
    )
    subparsers.add_parser("tree", help="Print the container hierarchy.")
    args = parser.parse_args()

    containers_dir = Path(__file__).resolve().parent.parent / "containers"
    if args.command == "closure":
        try:
            result: object = closure(containers_dir, args.containers, args.expand_parents, args.exclude)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result))
    elif args.command == "tree":
        print(tree(containers_dir))
    else:
        print(json.dumps(container_map(containers_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
