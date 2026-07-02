#!/usr/bin/env python3
"""
Check that Julia source files (.jl) carry the OSMC-PL 1.8 license header and an
up-to-date copyright year.

This is the Julia counterpart of OpenModelica's `check_runtime_license.py`. Julia
license headers live in a leading block comment `#= ... =#` (the OpenModelica
projects open it as `#= /* ... */ =#`). Only "normal" (non-runtime) headers are
relevant here, so the rules are:

  * the header must contain "This file is part of OpenModelica."
  * the header must contain "OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.8"
  * the copyright end-year must be the current year.

Usage:
  check_julia_license.py [--root ROOT] [--exceptions FILE]
                         [--update-year] [--fix-license] [--summary] DIRS...

Arguments:
  DIRS            One or more directories to check (relative to ROOT), e.g. `src`.
  --root ROOT     Repository root (default: directory of this script + "/../..").
  --exceptions FILE
                  Path to the exception list (default: next to this script,
                  julia-license-exceptions.txt). Files listed there are skipped —
                  use it for sources under a different license (e.g. vendored MIT
                  code).
  --update-year   Update copyright end-year to the current year.
  --fix-license   Replace wrong/missing headers with the correct OSMC-PL 1.8 one.
  --summary       Print a one-line summary even when there are no errors.

Exit codes:
  0  All files pass (or all failures were fixed with --fix-license).
  1  One or more files fail.

Exception-list format (one entry per line, evaluated in order, last match wins —
same semantics as .gitignore):
  # comments and blank lines are ignored
  path/relative/to/ROOT     exact file or directory (excluded)
  glob/pattern/**/*.jl       fnmatch glob relative to ROOT (excluded)
  !path/relative/to/ROOT    negation — re-includes a previously excluded path
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

CURRENT_YEAR: int = datetime.now().year

NORMAL_FILE_MARK = "This file is part of OpenModelica."

# "Copyright (c) YYYY" or "Copyright (c) YYYY-YYYY" (case-insensitive).
_COPYRIGHT_RE = re.compile(r"[Cc]opyright\s+\(c\)\s+(\d{4})(?:-(\d{4}))?", re.IGNORECASE)
# Exactly OSMC-PL version 1.8.
_OSMC_PL_1_8_RE = re.compile(r"OSMC PUBLIC LICENSE \(OSMC-PL\) VERSION 1\.8")
# Any OSMC-PL version.
_OSMC_PL_ANY_RE = re.compile(r"OSMC PUBLIC LICENSE \(OSMC-PL\)")

JULIA_EXTS = frozenset({".jl"})

# Canonical Julia OSMC-PL 1.8 header (matches the existing house style, which opens
# the Julia block comment and nests a C-style banner: `#= /* ... */ =#`).
OSMC_PL_1_8_LICENSE_TEXT_JL = f"""#= /*
* This file is part of OpenModelica.
*
* Copyright (c) 1998-{CURRENT_YEAR}, Open Source Modelica Consortium (OSMC),
* c/o Linköpings universitet, Department of Computer and Information Science,
* SE-58183 Linköping, Sweden.
*
* All rights reserved.
*
* THIS PROGRAM IS PROVIDED UNDER THE TERMS OF AGPL VERSION 3 LICENSE OR
* THIS OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.8.
* ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS PROGRAM CONSTITUTES
* RECIPIENT'S ACCEPTANCE OF THE OSMC PUBLIC LICENSE OR THE GNU AGPL
* VERSION 3, ACCORDING TO RECIPIENTS CHOICE.
*
* The OpenModelica software and the OSMC (Open Source Modelica Consortium)
* Public License (OSMC-PL) are obtained from OSMC, either from the above
* address, from the URLs:
* http://www.openmodelica.org or
* https://github.com/OpenModelica/ or
* http://www.ida.liu.se/projects/OpenModelica,
* and in the OpenModelica distribution.
*
* GNU AGPL version 3 is obtained from:
* https://www.gnu.org/licenses/licenses.html#GPL
*
* This program is distributed WITHOUT ANY WARRANTY; without
* even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE, EXCEPT AS EXPRESSLY SET FORTH
* IN THE BY RECIPIENT SELECTED SUBSIDIARY LICENSE CONDITIONS OF OSMC-PL.
*
* See the full OSMC Public License conditions for more details.
*
*/ =#"""

HEADER_READ_BYTES = 4096


def _file_ext(filename: str) -> str:
    return os.path.splitext(os.path.basename(filename))[1].lower()


def _is_license_block(block: str) -> bool:
    lower = block.lower()
    return "copyright" in lower or "osmc" in lower or "license" in lower


def extract_header(content: str) -> str:
    """Return the leading Julia `#= ... =#` license block, or leading `#` lines."""
    pos = 0
    while True:
        start = content.find("#=", pos)
        if start == -1:
            break
        end = content.find("=#", start + 2)
        if end == -1:
            break
        block = content[start : end + 2]
        if _is_license_block(block):
            return block
        pos = end + 2
    # Fallback: a run of leading `#` line comments.
    lines: list[str] = []
    for line in content.splitlines():
        s = line.strip()
        if s.startswith("#") or s == "":
            lines.append(line)
        else:
            break
    return "\n".join(lines)


def load_exceptions(exc_path: str | None) -> list[str]:
    if not exc_path or not os.path.exists(exc_path):
        return []
    patterns: list[str] = []
    with open(exc_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns


def _matches_pattern(rel: str, pattern: str) -> bool:
    pat = pattern.rstrip("/")
    if rel == pat or rel.startswith(pat + "/"):
        return True
    if fnmatch.fnmatch(rel, pattern):
        return True
    if fnmatch.fnmatch(os.path.basename(rel), pattern):
        return True
    return False


def is_excluded(rel_path: str, patterns: Iterable[str]) -> bool:
    rel = rel_path.replace(os.sep, "/")
    excluded = False
    for pattern in patterns:
        if pattern.startswith("!"):
            if _matches_pattern(rel, pattern[1:]):
                excluded = False
        elif _matches_pattern(rel, pattern):
            excluded = True
    return excluded


def _replace_license_header(filepath: str, content: str) -> bool:
    """Replace a leading `#= ... =#` license block, or prepend the header."""
    new_header = OSMC_PL_1_8_LICENSE_TEXT_JL.strip()
    pos = 0
    while True:
        start = content.find("#=", pos)
        if start == -1:
            break
        end = content.find("=#", start + 2)
        if end == -1:
            break
        block = content[start : end + 2]
        if _is_license_block(block):
            before = content[:start]
            after = content[end + 2 :]
            remaining = (before + after).lstrip("\n")
            new_content = new_header + "\n\n" + remaining
            with open(filepath, "w", encoding="utf-8") as fh:
                fh.write(new_content)
            return True
        pos = end + 2
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(new_header + "\n\n" + content)
    return True


def _update_copyright_year(filepath: str, content: str) -> bool:
    m = _COPYRIGHT_RE.search(content)
    if not m:
        return False
    start_year = m.group(1)
    end_year = int(m.group(2) or m.group(1))
    if end_year == CURRENT_YEAR:
        return False
    new_text = f"Copyright (c) {start_year}-{CURRENT_YEAR}"
    new_content = content[: m.start()] + new_text + content[m.end() :]
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(new_content)
    return True


def _copyright_year_errors(filepath: str, content: str, fix_year: bool) -> list[str]:
    m = _COPYRIGHT_RE.search(content)
    if not m:
        return ["copyright year not found"]
    end_year = int(m.group(2) or m.group(1))
    if end_year == CURRENT_YEAR:
        return []
    err = f"copyright year out of date ({end_year}, expected {CURRENT_YEAR})"
    if fix_year and _update_copyright_year(filepath, content):
        return [err + " [FIXED]"]
    return [err]


def check_file(filepath: str, fix_year: bool, fix_license: bool) -> list[str]:
    try:
        with open(filepath, encoding="utf-8", errors="replace") as fh:
            content = fh.read()
    except OSError as exc:
        return [f"cannot read file: {exc}"]

    header = extract_header(content[:HEADER_READ_BYTES])
    has_osmc_pl_1_8 = bool(_OSMC_PL_1_8_RE.search(header))
    has_osmc_pl_any = bool(_OSMC_PL_ANY_RE.search(header))
    has_normal_mark = NORMAL_FILE_MARK in header

    errors: list[str] = []
    if has_osmc_pl_1_8 and has_normal_mark:
        errors.extend(_copyright_year_errors(filepath, content, fix_year))
    else:
        if not has_osmc_pl_any:
            errors.append("missing OSMC-PL 1.8 license header")
        elif not has_osmc_pl_1_8:
            errors.append("wrong OSMC-PL version: expected 1.8")
        else:  # has 1.8 but not the "part of OpenModelica" mark
            errors.append('missing "This file is part of OpenModelica." mark')
        if fix_license and _replace_license_header(filepath, content):
            errors[-1] += " [FIXED]"
    return errors


def iter_source_files(root: Path, check_dirs: list[Path]) -> Iterable[Path]:
    for rel_dir in check_dirs:
        abs_dir = root.joinpath(rel_dir)
        if not abs_dir.is_dir():
            print(f"WARNING: directory not found: {abs_dir}", file=sys.stderr)
            continue
        for dirpath, dirnames, filenames in os.walk(abs_dir):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for fn in sorted(filenames):
                if _file_ext(fn) in JULIA_EXTS:
                    yield Path(dirpath).joinpath(fn)


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.normpath(os.path.join(script_dir, "..", ".."))
    default_exc = os.path.join(script_dir, "julia-license-exceptions.txt")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dirs", metavar="DIR", nargs="+", help="Directories to check, relative to ROOT.")
    parser.add_argument("--root", default=default_root, help="Repository root (default: %(default)s).")
    parser.add_argument("--exceptions", default=default_exc, metavar="FILE", help="Exception list file (default: %(default)s).")
    parser.add_argument("--update-year", action="store_true", help=f"Update copyright end-year to {CURRENT_YEAR}.")
    parser.add_argument("--fix-license", action="store_true", help="Replace wrong/missing headers with the OSMC-PL 1.8 header.")
    parser.add_argument("--summary", action="store_true", help="Always print a summary line.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    dirs = [Path(p) for p in args.dirs]
    exceptions = load_exceptions(args.exceptions)

    failures: list[tuple[str, list[str]]] = []
    checked = 0
    skipped = 0

    for abspath in iter_source_files(root, dirs):
        rel = str(abspath.relative_to(root)).replace(os.sep, "/")
        if is_excluded(rel, exceptions):
            skipped += 1
            continue
        checked += 1
        errs = check_file(str(abspath), args.update_year, args.fix_license)
        if errs:
            failures.append((rel, errs))

    unfixed_count = 0
    for rel, errs in failures:
        for err in errs:
            if err.endswith(" [FIXED]"):
                print(f"FIXED {rel}: {err[:-8]}")
            else:
                unfixed_count += 1
                print(f"FAIL  {rel}: {err}")

    if args.summary or failures:
        fixed_count = sum(1 for _, errs in failures for e in errs if e.endswith(" [FIXED]"))
        status = "PASSED" if unfixed_count == 0 else "FAILED"
        fix_note = f", {fixed_count} fixed" if fixed_count else ""
        print(f"\n{status}: checked {checked} files, skipped {skipped} (excluded), {unfixed_count} failures{fix_note}.")

    return 1 if unfixed_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
