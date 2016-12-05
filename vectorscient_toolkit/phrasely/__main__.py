"""
Command line interface for sentiment analysis algorithms.
"""
import argparse

from .factory import create_analyser_from_name, enumerate_analysers


parser = argparse.ArgumentParser(
    description="Sentiment Analysis command line interface.",
    prog="python -m phrasely")
available_analysers = [a.name() for a in enumerate_analysers()]
parser.add_argument("-n", "--name",
                    required=True,
                    default="weighted",
                    choices=available_analysers,
                    help="an analyser to be instantiated")
input_type = parser.add_mutually_exclusive_group(required=True)
input_type.add_argument("-s", "--string",
                        type=str,
                        help="string to be analysed")
input_type.add_argument("-f", "--file",
                        type=str,
                        help="path to a file with content to be analysed")
output = parser.add_mutually_exclusive_group(required=True)
output.add_argument("-o", "--output",
                    type=str,
                    help="path to an output file")
output.add_argument("-c", "--console",
                    action="store_true",
                    help="write sentiment result into standard output stream")

parser.add_argument("--format", choices=("numeric", "verbose", "both"),
                    required=True, default="numeric",
                    help="result format (score, sentiment string or both)")

args = parser.parse_args()
analyser = create_analyser_from_name(args.name)

if args.string:
    content = args.string
else:
    with open(args.file) as fp:
        content = fp.read()

score = analyser.sentiment(content)
verbose = analyser.verbose(score)
result = {
    "numeric": score,
    "verbose": verbose,
    "both": "{}::{}".format(score, verbose)
}[args.format]

if args.console:
    print(result)
else:
    with open(args.output, "wa") as fp:
        fp.write(result + "\n")
