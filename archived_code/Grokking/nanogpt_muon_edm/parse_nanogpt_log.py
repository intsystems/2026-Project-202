"""Parse a self-contained nanoGPT text log into analysis-ready tables.

The source log contains the complete training script before the actual run.
Consequently, this parser only accepts anchored, machine-readable records that
occur after RUNMETA.  The input file is never modified.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


META_RE = re.compile(r"^RUNMETA\s+(\{.*\})\s*$")
END_RE = re.compile(r"^RUNEND\s+(\{.*\})\s*$")
TRAIN_RE = re.compile(r"^step:(\d+)\s+train_loss:([-+0-9.eE]+)\s*$")
VAL_RE = re.compile(
    r"^step:(\d+)/(\d+)\s+val_loss:([-+0-9.eE]+)\s+"
    r"(?:val_loss_W:([-+0-9.eE]+)\s+)?"
    r"train_time:([-+0-9.eE]+)ms\s+step_avg:([-+0-9.eE]+)ms\s*$"
)
PROGRESS_RE = re.compile(
    r"^step:(\d+)/(\d+)\s+train_time:([-+0-9.eE]+)ms\s+"
    r"step_avg:([-+0-9.eE]+)ms\s*$"
)


def parse_log(path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta: dict | None = None
    run_end: dict | None = None
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    progress_rows: list[dict] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")

            match = META_RE.match(line)
            if match:
                if meta is not None:
                    raise ValueError("The file contains more than one RUNMETA record")
                meta = json.loads(match.group(1))
                meta["runmeta_line"] = line_number
                continue

            # Ignore the embedded source code and environment dump.
            if meta is None:
                continue

            match = TRAIN_RE.match(line)
            if match:
                train_rows.append(
                    {"step": int(match.group(1)), "train_loss_sum": float(match.group(2))}
                )
                continue

            match = VAL_RE.match(line)
            if match:
                val_rows.append(
                    {
                        "step": int(match.group(1)),
                        "train_steps": int(match.group(2)),
                        "val_loss": float(match.group(3)),
                        "val_loss_W": float(match.group(4)) if match.group(4) is not None else None,
                        "train_time_ms": float(match.group(5)),
                        "step_avg_ms": float(match.group(6)),
                    }
                )
                continue

            match = PROGRESS_RE.match(line)
            if match:
                progress_rows.append(
                    {
                        "step": int(match.group(1)),
                        "train_steps": int(match.group(2)),
                        "train_time_ms": float(match.group(3)),
                        "step_avg_ms": float(match.group(4)),
                    }
                )
                continue

            match = END_RE.match(line)
            if match:
                run_end = json.loads(match.group(1))

    if meta is None:
        raise ValueError("RUNMETA was not found")
    if run_end is None:
        raise ValueError("RUNEND was not found; the run may be incomplete")

    train = pd.DataFrame(train_rows).sort_values("step").reset_index(drop=True)
    validation = pd.DataFrame(val_rows).sort_values("step").reset_index(drop=True)
    progress = pd.DataFrame(progress_rows).sort_values("step").reset_index(drop=True)

    expected_steps = int(meta["train_steps"])
    expected = list(range(expected_steps))
    if train["step"].tolist() != expected:
        missing = sorted(set(expected) - set(train["step"].tolist()))
        duplicates = train.loc[train["step"].duplicated(), "step"].tolist()
        raise ValueError(f"Non-contiguous train log; missing={missing[:20]}, duplicates={duplicates[:20]}")

    world_size = int(meta["world_size"])
    tokens_per_step = int(meta["tokens_per_step"])
    local_tokens = tokens_per_step / world_size
    # The training code uses reduction='sum', then averages rank-local sums.
    train["train_loss_per_token"] = train["train_loss_sum"] / local_tokens

    num_iterations = int(meta["num_iterations"])
    cooldown_start = num_iterations * (1.0 - 0.45)
    x = (train["step"] / num_iterations).clip(upper=0.9999)
    train["lr_multiplier"] = 1.0
    cooldown = x >= (1.0 - 0.45)
    w = (1.0 - x[cooldown]) / 0.45
    train.loc[cooldown, "lr_multiplier"] = 0.1 + 0.9 * w
    train["phase"] = "constant_lr"
    train.loc[train["step"] >= cooldown_start, "phase"] = "lr_cooldown"
    train.loc[train["step"] >= num_iterations, "phase"] = "extension"

    metadata = dict(meta)
    metadata["run_end"] = run_end
    metadata["source_file"] = str(path.resolve())
    metadata["parsed_train_points"] = int(len(train))
    metadata["parsed_validation_points"] = int(len(validation))
    metadata["train_loss_normalization"] = {
        "reduction": "rank-local token sum followed by a mean across ranks",
        "local_tokens": local_tokens,
        "formula": "train_loss_per_token = train_loss_sum / (tokens_per_step / world_size)",
    }
    return metadata, train, validation, progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    metadata, train, validation, progress = parse_log(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output_dir / "train_log.csv", index=False)
    validation.to_csv(args.output_dir / "validation_log.csv", index=False)
    progress.to_csv(args.output_dir / "progress_log.csv", index=False)
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(
        f"Parsed {len(train)} train points and {len(validation)} validation points "
        f"for {metadata['run_id']}"
    )


if __name__ == "__main__":
    main()
