# coding=utf-8
# Created by bugface (https://github.com/bugface)
# First created at 3/3/22

import argparse
import traceback
from common_utils.output_format_converter import biaffine2brat, biaffine2bio


def main(args):
    if args.do_format == "brat":
        # to brat
        biaffine2brat(
            args.raw_input_dir_or_file,
            args.biaffine_output_file,
            args.mapping_file,
            args.formatted_output_dir,
            args.do_copy_raw_text)
    elif args.do_format == "bio":
        # to bio
        biaffine2bio(args.raw_input_dir_or_file, args.biaffine_output_file, args.formatted_output_dir)
    else:
        raise RuntimeError("we only support brat or bio format.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_input_dir_or_file", type=str, required=True,
                        help="The input text data directory. "
                             "For brat, we need raw text;"
                             "For bio we need raw bio data file")
    parser.add_argument("--biaffine_output_file", type=str, required=True,
                        help="The output data directory from biaffine.")
    parser.add_argument("--mapping_file", type=str, default=None,
                        help="only required for brat format where we stored the original offset information")
    parser.add_argument("--formatted_output_dir", type=str, required=True,
                        help="The final output data directory.")
    parser.add_argument("--do_format", type=str, default="bio",
                        help="bio or brat")
    parser.add_argument("--do_copy_raw_text", type=bool, default=False,
                        help="copy raw text to output; only required for brat")
    global_args = parser.parse_args()

    if global_args.do_format == 1 and not global_args.mapping_file:
        raise RuntimeError("you need to provide the mapping file since you want to convert to brat format")

    main(global_args)
