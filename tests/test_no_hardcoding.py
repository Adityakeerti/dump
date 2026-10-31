"""Test AST scanning of src/ to enforce no hardcoded numeric or path literals."""

import ast
import os
import re
from pathlib import Path
import pytest


ALLOWED_NUMBERS = {-1, 0, 1, 2, -1.0, 0.0, 1.0, 2.0}


def is_path_or_url(s: str) -> bool:
    if not isinstance(s, str) or len(s) == 0:
        return False
    # Check for URL
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", s):
        return True
    # Check for file path with filename and extension (e.g. data.csv, config/app.yaml)
    if re.search(r"[a-zA-Z0-9_-]+\.(yaml|yml|json|csv|png|html|txt|h5|pt|pkl|parquet)$", s, re.IGNORECASE):
        return True
    # Check for absolute or relative unix/windows paths (e.g. /etc/hosts, ./dir, ../dir)
    if s.startswith("/") or s.startswith("./") or s.startswith("../") or s.startswith("..\\") or s.startswith(".\\"):
        return True
    # Check for slash separating path components without whitespace or format brackets
    if "/" in s and not (" " in s or "\n" in s or "{" in s or "}" in s):
        return True
    return False


def get_docstrings(tree: ast.AST) -> set:
    """Collect all docstring nodes in the AST to exclude them."""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                docstrings.add(id(node.body[0].value))
    return docstrings


def scan_file_for_hardcoding(filepath: Path) -> list[str]:
    """Scans a single python file and returns violation messages."""
    violations = []
    is_units_file = filepath.name == "units.py"

    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        return [f"Syntax error in {filepath}: {e}"]

    docstring_ids = get_docstrings(tree)

    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            # Check UnaryOp for negative numbers like -1, -5
            if isinstance(child, ast.UnaryOp) and isinstance(child.op, ast.USub) and isinstance(child.operand, ast.Constant):
                val = child.operand.value
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    neg_val = -val
                    if not is_units_file and neg_val not in ALLOWED_NUMBERS:
                        violations.append(
                            f"{filepath}:{child.lineno}: Disallowed numeric literal: {neg_val}"
                        )
                continue

            # Standard constants
            if isinstance(child, ast.Constant):
                # Skip if part of a UnaryOp parent already handled
                if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.USub):
                    continue

                # Skip docstrings
                if id(child) in docstring_ids:
                    continue

                val = child.value
                # Check numeric literals
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    if not is_units_file and val not in ALLOWED_NUMBERS:
                        violations.append(
                            f"{filepath}:{child.lineno}: Disallowed numeric literal: {val}"
                        )

                # Check string literals for paths/URLs
                elif isinstance(val, str):
                    if is_path_or_url(val):
                        violations.append(
                            f"{filepath}:{child.lineno}: Disallowed hardcoded path or URL: {val!r}"
                        )

    return violations


def test_no_hardcoding_in_src():
    """Verify that all source files in src/ contain no disallowed numeric, path, or URL literals."""
    src_dir = Path("src")
    assert src_dir.exists(), "src/ directory does not exist"

    all_violations = []
    py_files = list(src_dir.glob("**/*.py"))
    assert len(py_files) > 0, "No python files found in src/"

    for py_file in py_files:
        violations = scan_file_for_hardcoding(py_file)
        all_violations.extend(violations)

    if all_violations:
        msg = f"Found {len(all_violations)} hardcoding violations:\n" + "\n".join(all_violations)
        pytest.fail(msg)
