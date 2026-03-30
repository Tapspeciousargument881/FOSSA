"""
zedd_check_format.py

Validates a user-submitted zip file for ZEDD evaluation before running metrics.
Run this script to catch formatting mistakes early.

Usage:
    python zedd_check_format.py --zip path/to/zedd_outputs.zip
"""

import os
import sys
import zipfile
import argparse
import numpy as np

EXPECTED_COUNT = 50
EXPECTED_W = 1824
EXPECTED_H = 1216
EXPECTED_SHAPE = (EXPECTED_H, EXPECTED_W)


def expected_name(i: int) -> str:
    return f"zedd_output_{i:04d}.npy"


def check_zip(zip_path: str) -> bool:
    errors = []
    warnings = []

    # ------------------------------------------------------------------ #
    # 1. File existence and extension
    # ------------------------------------------------------------------ #
    if not os.path.exists(zip_path):
        print(f"[ERROR] File not found: {zip_path}")
        return False

    if not zip_path.endswith(".zip"):
        errors.append(f"Input file does not have a .zip extension: '{zip_path}'")

    if not zipfile.is_zipfile(zip_path):
        errors.append(f"File is not a valid zip archive: '{zip_path}'")
        _report(errors, warnings)
        return False

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()

        # ------------------------------------------------------------------ #
        # 2. No subdirectories — all files must be at the root level
        # ------------------------------------------------------------------ #
        subdirs = [n for n in all_names if n.endswith("/") or "/" in n]
        if subdirs:
            errors.append(
                f"Zip contains subdirectories or nested paths. All .npy files must be "
                f"stored at the root level (no folders). Offending entries:\n"
                + "\n".join(f"    {s}" for s in subdirs)
            )

        # ------------------------------------------------------------------ #
        # 3. Only .npy files are allowed
        # ------------------------------------------------------------------ #
        non_npy = [n for n in all_names if not n.endswith(".npy") and not n.endswith("/")]
        if non_npy:
            errors.append(
                f"Zip contains non-.npy files:\n"
                + "\n".join(f"    {n}" for n in non_npy)
            )

        npy_names = [n for n in all_names if n.endswith(".npy")]

        # ------------------------------------------------------------------ #
        # 4. Correct number of files
        # ------------------------------------------------------------------ #
        if len(npy_names) != EXPECTED_COUNT:
            errors.append(
                f"Expected exactly {EXPECTED_COUNT} .npy files, found {len(npy_names)}."
            )

        # ------------------------------------------------------------------ #
        # 5. All expected filenames present
        # ------------------------------------------------------------------ #
        expected_names = {expected_name(i) for i in range(1, EXPECTED_COUNT + 1)}
        present_names = set(npy_names)

        missing = sorted(expected_names - present_names)
        if missing:
            errors.append(
                f"{len(missing)} expected file(s) are missing:\n"
                + "\n".join(f"    {n}" for n in missing)
            )

        extra = sorted(present_names - expected_names)
        if extra:
            errors.append(
                f"{len(extra)} unexpected file(s) found in the zip:\n"
                + "\n".join(f"    {n}" for n in extra)
            )

        # ------------------------------------------------------------------ #
        # 6. Per-file content checks (only for files that match the naming)
        # ------------------------------------------------------------------ #
        checkable = sorted(present_names & expected_names)
        for name in checkable:
            try:
                with zf.open(name) as f:
                    arr = np.load(f)
            except Exception as e:
                errors.append(f"'{name}': failed to load as numpy array — {e}")
                continue

            # Shape
            if arr.shape != EXPECTED_SHAPE:
                errors.append(
                    f"'{name}': wrong shape {arr.shape}. "
                    f"Expected (H={EXPECTED_H}, W={EXPECTED_W}) = {EXPECTED_SHAPE}. "
                    f"Note: arrays must be 2-D (H, W); do not include a channel dimension."
                )

            # dtype — should be float-compatible
            if not np.issubdtype(arr.dtype, np.floating):
                warnings.append(
                    f"'{name}': dtype is {arr.dtype}, expected a float type (e.g. float32). "
                    f"Values will be cast automatically but this may indicate an error."
                )

            # Finite values
            n_nan = int(np.sum(np.isnan(arr)))
            n_inf = int(np.sum(np.isinf(arr)))
            if n_nan > 0:
                errors.append(
                    f"'{name}': contains {n_nan} NaN value(s). "
                    f"Predicted depth must be fully finite."
                )
            if n_inf > 0:
                errors.append(
                    f"'{name}': contains {n_inf} Inf value(s). "
                    f"Predicted depth must be fully finite."
                )


    return _report(errors, warnings)


def _report(errors: list, warnings: list) -> bool:
    if warnings:
        print(f"\n{'='*60}")
        print(f"WARNINGS ({len(warnings)})")
        print(f"{'='*60}")
        for w in warnings:
            print(f"  [WARNING] {w}")

    if errors:
        print(f"\n{'='*60}")
        print(f"ERRORS ({len(errors)})")
        print(f"{'='*60}")
        for e in errors:
            print(f"  [ERROR] {e}")
        print(f"\n{'='*60}")
        print("Format check FAILED. Please fix the errors above before submitting.")
        print(f"{'='*60}\n")
        return False
    else:
        print(f"\n{'='*60}")
        if warnings:
            print("Format check PASSED with warnings (see above).")
        else:
            print("Format check PASSED. Your zip file looks correct!")
        print(
            f"  - {EXPECTED_COUNT} files, all correctly named\n"
            f"  - All arrays have shape ({EXPECTED_H}, {EXPECTED_W})\n"
            f"  - No NaN or Inf values detected"
        )
        print(f"{'='*60}\n")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate a ZEDD submission zip file before running evaluation."
    )
    parser.add_argument(
        "--zip",
        type=str,
        required=True,
        help="Path to the zip file to validate.",
    )
    args = parser.parse_args()

    ok = check_zip(args.zip)
    sys.exit(0 if ok else 1)
