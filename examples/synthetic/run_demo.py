"""Generate synthetic inputs and run the framework demonstration."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import chacam
from chacam.synthetic import make_synthetic, write_synthetic


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    work_dir = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else script_dir
    config_name = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
    data = make_synthetic(seed=0, n_samples=10)
    write_synthetic(data, work_dir / "data")
    config_path = work_dir / config_name
    if work_dir != script_dir:
        shutil.copyfile(script_dir / config_name, config_path)
    result = chacam.run(config_path)
    print("status:")
    print(result.predictions["status"].value_counts(sort=False).to_string())
    print("unresolved_reason:")
    print(result.predictions["unresolved_reason"].value_counts(dropna=False, sort=False).to_string())
    metrics = json.loads((result.output_dir / "metrics.json").read_text(encoding="utf-8"))
    print("metrics:")
    for name in ("stage1_accuracy", "resolution_rate", "reference_agreement", "fine_accuracy", "named_accuracy"):
        print(f"  {name}: {metrics['pooled'][name]['value']}")
    print(f"output_dir: {result.output_dir}")


if __name__ == "__main__":
    main()
