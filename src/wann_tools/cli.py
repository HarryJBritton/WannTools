
import argparse
import sys
import logging

from wann_tools.config import load_config
from wann_tools.runner import run
from wann_tools.estimator import estimate


def configure_logging(config):
    level = (
        config.debug.log_level.upper()
        if config.debug.enabled
        else "INFO"
    )

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

def main():
    parser = argparse.ArgumentParser(
        prog="wannrun",
        description="Run wann_tools calculations.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    for command in ("validate", "estimate", "run"):
        subparser = subparsers.add_parser(
            command,
            help=f"{command.capitalize()} the calculation configuration",
        )
        subparser.add_argument(
            "config",
            help="Path to the YAML configuration file",
        )

    args = parser.parse_args()

    try:
        config = load_config(args.config)
        configure_logging(config)

        if args.command == "validate":
            print(f"Configuration is valid: {args.config}")

        elif args.command == "estimate":
            report = estimate(config)
            print(report)

        elif args.command == "run":
            result = run(config)
            print("Calculation completed successfully.")

    except (ValueError, OSError, NotImplementedError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
