"""Generate synthetic inputs and run the M2 demonstration."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import chacam
from chacam.synthetic import make_synthetic, write_synthetic


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    work_dir = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else script_dir
    data = make_synthetic(seed=0, n_samples=10)
    write_synthetic(data, work_dir / "data")
    config_path = work_dir / "config.yaml"
    if work_dir != script_dir:
        shutil.copyfile(script_dir / "config.yaml", config_path)
    result = chacam.run(config_path)
    print("status:")
    print(result.predictions["status"].value_counts(sort=False).to_string())
    print("unresolved_reason:")
    print(result.predictions["unresolved_reason"].value_counts(dropna=False, sort=False).to_string())
    print(f"output_dir: {result.output_dir}")


if __name__ == "__main__":
    main()
