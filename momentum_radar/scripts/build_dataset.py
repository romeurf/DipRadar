"""momentum_radar/scripts/build_dataset.py — RETIRED.

Built the dataset that fed MomentumRadar's ML trainer (also retired).
Both were anti-predictive (CV IC -0.0042). The dataset has no consumer
now — keeping this file would just produce a parquet that nothing reads.

Stubbed for import compatibility. Calling main() raises with a clear
retirement message.
"""
from __future__ import annotations


def main(*args, **kwargs):
    raise RuntimeError(
        "MomentumRadar dataset builder is RETIRED alongside the trainer "
        "(anti-predictive, CV IC = -0.0042). The rule-based scanner at "
        "momentum_radar/scanner.py needs no dataset and is the only "
        "shipped momentum component."
    )


if __name__ == "__main__":
    main()
