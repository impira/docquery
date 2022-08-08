import argparse
import logging
import sys


def main(args=None):
    """The main routine."""
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(add_help=False, description="docqa is a cli tool to work with documents.")
    parser.add_argument("--verbose", "-v", default=False, action="store_true")

    args = parser.parse_args(args=args)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=level)
    logging.info("Hello world!")


if __name__ == "__main__":
    sys.exit(main())
