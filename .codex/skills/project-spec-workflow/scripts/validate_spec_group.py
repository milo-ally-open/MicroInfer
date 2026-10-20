#!/usr/bin/env python3
import re
import sys
from pathlib import Path


REQUIRED = ("proposal.md",)
PLACEHOLDER = re.compile(r"<(?!br\s*/?>)[^>\n]+>|YYYY-MM-DD|简短名称", re.IGNORECASE)
REQ_DEF = re.compile(r"^###\s+(REQ-\d+)\b", re.MULTILINE)
AC_DEF = re.compile(r"\*\*(AC-\d+\.\d+)\*\*")
TASK_DEF = re.compile(r"^-\s+\[[ xX]\]\s+\*\*(T-\d+)\b[^\n]*", re.MULTILINE)
REQ_REF = re.compile(r"\bREQ-\d+\b")
AC_REF = re.compile(r"\bAC-\d+\.\d+\b")


def duplicates(values):
    seen = set()
    repeated = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return sorted(repeated)


def main():
    if len(sys.argv) != 2:
        print("Usage: validate_spec_group.py <Specification/domain-group>", file=sys.stderr)
        return 2

    group = Path(sys.argv[1]).resolve()
    if not group.is_dir():
        print("FAIL: group directory does not exist: " + str(group))
        return 1

    errors = []
    documents = {}
    for name in ("proposal.md", "spec.md", "design.md", "tasks.md"):
        path = group / name
        if path.is_file():
            documents[name] = path.read_text(encoding="utf-8")
    for name in REQUIRED:
        if name not in documents:
            errors.append("missing required document: " + name)
    if not documents:
        errors.append("group contains no supported documents")

    for name, content in documents.items():
        if not content.strip():
            errors.append(name + " is empty")
        for marker in sorted(set(PLACEHOLDER.findall(content))):
            errors.append(name + " has an unfinished template marker: " + marker)

    requirements = []
    acceptances = []
    if "spec.md" in documents:
        spec = documents["spec.md"]
        requirements = REQ_DEF.findall(spec)
        acceptances = AC_DEF.findall(spec)
        if not requirements:
            errors.append("spec.md defines no ### REQ-xx requirements")
        if not acceptances:
            errors.append("spec.md defines no bold AC-xx.n acceptance IDs")
        for identifier in duplicates(requirements):
            errors.append("duplicate requirement definition: " + identifier)
        for identifier in duplicates(acceptances):
            errors.append("duplicate acceptance definition: " + identifier)
        requirement_numbers = {value.split("-")[1] for value in requirements}
        for identifier in acceptances:
            if identifier.split("-")[1].split(".")[0] not in requirement_numbers:
                errors.append("acceptance has no matching requirement: " + identifier)

    if "tasks.md" in documents:
        tasks = documents["tasks.md"]
        task_lines = TASK_DEF.findall(tasks)
        if not task_lines:
            errors.append("tasks.md defines no checkbox tasks with T-xx IDs")
        for identifier in duplicates(task_lines):
            errors.append("duplicate task definition: " + identifier)
        if requirements and acceptances:
            defined_reqs = set(requirements)
            defined_acs = set(acceptances)
            mapped_reqs = set()
            mapped_acs = set()
            for line in TASK_DEF.finditer(tasks):
                task_line = line.group(0)
                req_refs = set(REQ_REF.findall(task_line))
                ac_refs = set(AC_REF.findall(task_line))
                unknown_reqs = req_refs - defined_reqs
                unknown_acs = ac_refs - defined_acs
                for identifier in sorted(unknown_reqs):
                    errors.append(line.group(1) + " refers to undefined requirement " + identifier)
                for identifier in sorted(unknown_acs):
                    errors.append(line.group(1) + " refers to undefined acceptance " + identifier)
                if not req_refs and not ac_refs:
                    errors.append(line.group(1) + " is not linked to a REQ or AC")
                mapped_reqs.update(req_refs & defined_reqs)
                mapped_acs.update(ac_refs & defined_acs)
            for identifier in sorted(defined_reqs - mapped_reqs):
                errors.append("requirement is not covered by a task: " + identifier)
            for identifier in sorted(defined_acs - mapped_acs):
                errors.append("acceptance is not covered by a task: " + identifier)

    if errors:
        print("FAIL: " + str(len(errors)) + " issue(s) in " + str(group))
        for error in errors:
            print("  - " + error)
        return 1

    print("PASS: " + str(group))
    print("  documents: " + ", ".join(sorted(documents)))
    if requirements:
        print("  requirements: " + str(len(requirements)) + "; acceptance criteria: " + str(len(acceptances)))
    if "tasks.md" in documents:
        print("  tasks: " + str(len(TASK_DEF.findall(documents["tasks.md"]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
