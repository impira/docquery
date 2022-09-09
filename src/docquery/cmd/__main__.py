import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import logging
import sys
import textwrap

import transformers

from ..transformers_patch import PIPELINE_DEFAULTS


_module_not_found_error = None
try:
    from . import scan
except ModuleNotFoundError as e:
    _module_not_found_error = e

if _module_not_found_error is not None:
    raise ModuleNotFoundError(
        textwrap.dedent(
            f"""\
            At least one dependency not found: {str(_module_not_found_error)!r}
            It is possible that docquery was installed without the CLI dependencies. Run:

              pip install 'docquery[cli]'

            to install impira with the CLI dependencies."""
        )
    )


def main(args=None):
    """The main routine."""
    if args is None:
        args = sys.argv[1:]

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--verbose", "-v", default=False, action="store_true")
    parent_parser.add_argument(
        "--checkpoint",
        default=None,
        help=f"A custom model checkpoint to use (other than {PIPELINE_DEFAULTS['document-question-answering']})",
    )

    parser = argparse.ArgumentParser(description="docquery is a cli tool to work with documents.")
    subparsers = parser.add_subparsers(help="sub-command help", dest="subcommand", required=True)

    for module in [scan]:
        module.build_parser(subparsers, parent_parser)

    args = parser.parse_args(args=args)
    level = logging.DEBUG if args.verbose else logging.INFO
    if not args.verbose:
        transformers.logging.set_verbosity_error()
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=level)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
