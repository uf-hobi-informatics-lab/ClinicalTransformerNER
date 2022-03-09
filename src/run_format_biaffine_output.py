# coding=utf-8
# Created by bugface (https://github.com/bugface)
# First created at 3/3/22

import argparse
import traceback

from common_utils.output_format_converter import biaffine2brat, biaffine2bio


def main(args):
    if args.do_format == 1:
        # to brat
        biaffine2brat()
    else:
        # to bio
        biaffine2bio()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_input_file", type=str, required=True,
                        help="The input data directory.")
    parser.add_argument("--biaffine_output_file", type=str, required=True,
                        help="The output data directory from biaffine.")
    parser.add_argument("--mapping_file", type=str,
                        help="The output data directory.")
    parser.add_argument("--formatted_output_dir", type=str, required=True,
                        help="The final output data directory.")
    parser.add_argument("--do_format", default=0, type=int,
                        help="0=bio; 1=brat")
    global_args = parser.parse_args()

    main(global_args)
