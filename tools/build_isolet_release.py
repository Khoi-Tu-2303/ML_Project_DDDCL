"""Build the canonical full ISOLET (7797 x 617, 26 classes) from the UCI archive isolet.zip.

Reads the zip in memory (nothing is extracted to disk), checks the archive SHA-256, decodes the
compress(1) ``.Z`` members with a pure-Python LZW decoder, and writes ``data/isolet.csv``,
``provenance/isolet.json`` and the ``isolet`` manifest entry. Row order = UCI order:
``isolet1+2+3+4`` (6238 rows) followed by ``isolet5`` (1559 rows). Nothing is scaled, split,
imputed or resampled. Raw labels 1..26 map to 0..25.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

UCI_URL = "https://archive.ics.uci.edu/static/public/54/isolet.zip"
UCI_PAGE = "https://archive.ics.uci.edu/dataset/54/isolet"
ARCHIVE_SHA256 = "fe2e0d45f1057d112e051d309070992fc178e7f6922cc7b4e8b2bd310303a053"
MEMBERS = ["isolet1+2+3+4.data.Z", "isolet5.data.Z"]
N_FEATURES, N_CLASSES, N_ROWS = 617, 26, 7797
NAME = "isolet"


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def unlzw(data: bytes) -> bytes:
    """Decode Unix compress(1) ``.Z`` data (LZW, variable 9..maxbits codes, optional block mode)."""
    if data[:2] != b"\x1f\x9d":
        raise ValueError("not a compress(1) .Z stream")
    maxbits, block = data[2] & 0x1F, bool(data[2] & 0x80)
    if not 9 <= maxbits <= 16:
        raise ValueError(f"unsupported max code size {maxbits}")
    nbits, maxcode, free = 9, (1 << 9) - 1, 257 if block else 256
    prefix, suffix = [0] * (1 << maxbits), list(range(256)) + [0] * ((1 << maxbits) - 256)
    out = bytearray()
    pos, group_bytes, off, clear = 3, 0, 0, False
    old, fin = -1, 0
    total = len(data)
    while True:
        if clear or free > maxcode or off + nbits > group_bytes * 8:
            if clear:
                nbits, maxcode, clear = 9, (1 << 9) - 1, False
            elif free > maxcode:
                nbits += 1
                maxcode = (1 << nbits) - 1 if nbits < maxbits else 1 << maxbits
            pos += group_bytes
            group_bytes, off = min(nbits, total - pos), 0
            if group_bytes <= 0 or off + nbits > group_bytes * 8:
                break
        bit = pos * 8 + off
        chunk = int.from_bytes(data[bit >> 3:(bit >> 3) + 3], "little")
        code = (chunk >> (bit & 7)) & ((1 << nbits) - 1)
        off += nbits
        if old == -1:
            fin = old = code
            out.append(fin)
            continue
        if block and code == 256:
            free, clear, old = 257, True, -1
            continue
        incode, stack = code, []
        if code >= free:
            stack.append(fin)
            code = old
        while code >= 256:
            stack.append(suffix[code])
            code = prefix[code]
        fin = suffix[code]
        stack.append(fin)
        out.extend(reversed(stack))
        if free < (1 << maxbits):
            prefix[free], suffix[free] = old, fin
            free += 1
        old = incode
    return bytes(out)


def parse(blob: bytes, start: int) -> list[list[str]]:
    """Return canonical rows [row_id, 617 raw feature strings, label] for one decoded member."""
    rows = []
    for line in blob.decode("ascii").splitlines():
        if not line.strip():
            continue
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != N_FEATURES + 1:
            raise ValueError(f"row {start + len(rows)}: {len(fields)} fields, expected {N_FEATURES + 1}")
        if not all(math.isfinite(float(v)) for v in fields[:-1]):
            raise ValueError(f"row {start + len(rows)}: non-finite feature")
        raw = float(fields[-1])
        if raw != int(raw) or not 1 <= int(raw) <= N_CLASSES:
            raise ValueError(f"row {start + len(rows)}: bad label {fields[-1]!r}")
        rows.append([f"{NAME}:{start + len(rows):06d}", *fields[:-1], str(int(raw) - 1)])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path.home() / "Downloads" / "isolet.zip")
    parser.add_argument("--release", type=Path, default=Path("hf_release"))
    args = parser.parse_args()

    blob = args.zip.read_bytes()
    if sha256_bytes(blob) != ARCHIVE_SHA256:
        raise ValueError("isolet.zip SHA-256 differs from the pinned UCI archive; investigate, do not accept")
    rows, member_info = [], {}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for member in MEMBERS:
            raw_z = z.read(member)
            decoded = unlzw(raw_z)
            part = parse(decoded, len(rows))
            member_info[member.removesuffix(".Z")] = {
                "z_sha256": sha256_bytes(raw_z), "decoded_sha256": sha256_bytes(decoded),
                "first_row": len(rows), "n_rows": len(part)}
            rows += part
    counts = Counter(r[-1] for r in rows)
    class_counts = {str(c): counts[str(c)] for c in range(N_CLASSES)}
    if len(rows) != N_ROWS or len(counts) != N_CLASSES:
        raise ValueError(f"unexpected shape: {len(rows)} rows, {len(counts)} classes")

    features = [f"feature_{i + 1}" for i in range(N_FEATURES)]
    out = args.release / "data" / f"{NAME}.csv"
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["row_id", *features, "label"])
        writer.writerows(rows)

    label_mapping = {str(i + 1): i for i in range(N_CLASSES)}
    label_meanings = {str(i): f"letter_{chr(ord('A') + i)}" for i in range(N_CLASSES)}
    provenance = {
        "url": UCI_URL, "page": UCI_PAGE, "archive_sha256": ARCHIVE_SHA256, "members": member_info,
        "n_rows": len(rows), "features": features, "class_counts": class_counts,
        "label_mapping": label_mapping, "label_meanings": label_meanings, "license": "CC-BY-4.0",
        "row_order": "isolet1+2+3+4 (rows 0-6237) then isolet5 (rows 6238-7796); UCI order preserved",
        "transforms": ["decode compress(1) .Z", "strip whitespace", "stable row_id",
                       "raw label k (1..26) -> k-1", "generic feature names"],
        "no_scaling_no_splitting_no_resampling": True,
    }
    write_json(args.release / "provenance" / f"{NAME}.json", provenance)
    entry = {
        "filename": f"data/{NAME}.csv", "sha256": sha256_bytes(out.read_bytes()), "n_rows": len(rows),
        "feature_columns": features, "label_column": "label", "row_id_column": "row_id",
        "class_counts": class_counts, "label_mapping": label_mapping, "label_meanings": label_meanings,
        "source_url": UCI_URL, "source_sha256": ARCHIVE_SHA256, "provenance_file": f"provenance/{NAME}.json",
        "license": "CC-BY-4.0", "status": "verified_primary_source",
    }
    manifest_path = args.release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["datasets"][NAME] = entry
    write_json(manifest_path, manifest)
    print(json.dumps({"rows": len(rows), "features": N_FEATURES, "class_counts": class_counts,
                      "csv_sha256": entry["sha256"], "manifest_sha256": sha256_bytes(manifest_path.read_bytes())}, indent=2))


if __name__ == "__main__":
    main()
