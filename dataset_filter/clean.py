import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm


EXCLUDED_TYPES = {
    "samples",
    "events",
    "a11y",
    "master - b730ba8 prototype",
    "compat",
    "revert",
}

TYPE_MAP = {
    "refact": "refactor",
    "bug": "fix",
    "enhance": "feat",
    "examples": "docs",
    "demo": "docs",
    "cleanup": "chore",
    "amend": "chore",
}

pattern = re.compile(r"\s*\(#\d+(?:,\s*#\d+)*\)\s*$")


def clean_subject(subject: str) -> str:
    if not subject:
        return subject
    subject = re.sub(pattern, "", subject)
    return subject.strip()


def has_value(data: dict, key: str) -> bool:
    value = data.get(key)
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def should_keep(data: dict) -> bool:
    if not has_value(data, "type") or not has_value(data, "scope"):
        return False

    type_value = (
        data["type"].strip().lower()
        if isinstance(data["type"], str)
        else data["type"]
    )
    return type_value not in EXCLUDED_TYPES


def normalize_type(data: dict) -> None:
    type_value = (
        data["type"].strip().lower()
        if isinstance(data["type"], str)
        else data["type"]
    )
    data["type"] = TYPE_MAP.get(type_value, type_value)


def parse_args():
    parser = argparse.ArgumentParser(description="Clean and normalize a JSONL dataset.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:

        total = 0
        kept = 0
        removed = 0

        for line in tqdm(fin):
            total += 1
            data = json.loads(line)

            if not should_keep(data):
                removed += 1
                continue

            normalize_type(data)

            if "subject" in data:
                data["subject"] = clean_subject(data["subject"])

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Cleaned file saved to: {output_path}")
    print(f"Total: {total}, kept: {kept}, removed: {removed}")


if __name__ == "__main__":
    main()
