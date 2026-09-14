"""ICS AI testbench -- command line entry point.

    python app.py all         generate data, train both models, check specs, export
    python app.py generate    synthetic run-to-failure corpus only
    python app.py train       train N-HiTS and TFT
    python app.py evaluate    score both models against ICS1 / ICS2 / I1 / I3
    python app.py export      ONNX export + contract for the .NET host
    python app.py replay      stream payloads through both models, live

Run `python app.py <command> --help` for the options on each.
"""

from __future__ import annotations

import argparse
import sys
import warnings

import pandas as pd

from config import RunConfig

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _config(args: argparse.Namespace) -> RunConfig:
    cfg = RunConfig()
    if getattr(args, "runs", None):
        cfg.synth.n_runs = args.runs
    if getattr(args, "epochs", None):
        cfg.nhits.max_epochs = args.epochs
        cfg.tft.max_epochs = args.epochs
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed
        cfg.synth.seed = args.seed
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="Testbench for the two ICS-owned models: N-HiTS (RUL) and "
                    "TFT (fault classification). All data is synthetic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser, *, training: bool = False) -> None:
        sub.add_argument("--runs", type=int, help="number of synthetic runs")
        sub.add_argument("--seed", type=int, help="random seed")
        if training:
            sub.add_argument("--epochs", type=int, help="max epochs per model")

    gen = subparsers.add_parser("generate", help="build the synthetic corpus")
    add_common(gen)
    gen.add_argument("--force", action="store_true",
                     help="regenerate even if a corpus already exists")

    train_parser = subparsers.add_parser("train", help="train both models")
    add_common(train_parser, training=True)
    train_parser.add_argument("--force", action="store_true",
                              help="regenerate the corpus first")
    train_parser.add_argument("--model", choices=["both", "nhits", "tft"],
                              default="both")

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="score both models against the specs")
    add_common(evaluate_parser)

    export_parser = subparsers.add_parser(
        "export", help="ONNX export + contract for the .NET host")
    add_common(export_parser)

    replay_parser = subparsers.add_parser(
        "replay", help="stream synthetic payloads through both models")
    add_common(replay_parser)
    replay_parser.add_argument("--run-id", type=int, default=None,
                               help="which test run to replay (default: first)")
    replay_parser.add_argument("--speed", type=float, default=0.0,
                               help="seconds to pause between windows")
    replay_parser.add_argument("--limit", type=int, default=60,
                               help="how many windows to replay")

    subparsers.add_parser(
        "selftest",
        help="check the payload codec, feature contract, bearing orders and "
             "DSP chain -- no model or downloads needed")

    datasets_parser = subparsers.add_parser(
        "datasets", help="list the public datasets and what each can support")
    datasets_parser.add_argument(
        "--load", choices=["femto", "cwru", "xjtu_sy"],
        help="convert one dataset into this project's feature space")
    datasets_parser.add_argument(
        "--mirror-plane", action="store_true",
        help="FABRICATES the second measurement plane for 2-channel datasets. "
             "Smoke-testing only -- never for a reported number.")

    all_parser = subparsers.add_parser(
        "all", help="generate, train, evaluate and export in one go")
    add_common(all_parser, training=True)
    all_parser.add_argument("--force", action="store_true",
                            help="regenerate the corpus first")

    args = parser.parse_args(argv)
    cfg = _config(args)

    if args.command == "generate":
        import train as train_mod
        train_mod.generate(cfg, force=args.force)
        return 0

    if args.command == "train":
        import train as train_mod
        train_mod.run(cfg, force=args.force, models=args.model)
        return 0

    if args.command == "evaluate":
        import evaluate as evaluate_mod
        evaluate_mod.run(cfg)
        return 0

    if args.command == "export":
        import export_onnx
        export_onnx.run(cfg)
        return 0

    if args.command == "selftest":
        import selftest
        return selftest.run()

    if args.command == "datasets":
        import external
        if not args.load:
            print(external.describe())
            return 0
        policy = (
            external.ChannelPolicy.MIRROR_PLANE if args.mirror_plane
            else external.ChannelPolicy.STRICT
        )
        frame = external.load(args.load, policy)
        if frame is None:
            return 1
        out = external.EXTERNAL_DIR / f"{args.load}_windows.parquet"
        frame.to_parquet(out, index=False)
        print(f"\n  {len(frame):,} windows -> {out}")
        if frame.get("band_was_narrowed", pd.Series([False])).any():
            print("  ! the 2-8 kHz resonance band did not fit under this "
                  "dataset's Nyquist limit and was narrowed. These features "
                  "are not directly comparable with the rig's.")
        return 0

    if args.command == "replay":
        import replay as replay_mod
        replay_mod.run(cfg, run_id=args.run_id, limit=args.limit, pause=args.speed)
        return 0

    if args.command == "all":
        import evaluate as evaluate_mod
        import export_onnx
        import train as train_mod

        train_mod.run(cfg, force=args.force)
        print()
        evaluate_mod.run(cfg)
        print()
        export_onnx.run(cfg)
        return 0

    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
