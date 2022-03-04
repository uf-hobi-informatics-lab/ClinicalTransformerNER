# coding=utf-8
# Created by bugface (https://github.com/bugface)
# First created at 3/3/22

import argparse
import traceback

from common_utils.output_format_converter import biaffine2brat, biaffine2bio


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_text_dir", type=str, required=True,
                        help="The input data directory.")
    parser.add_argument("--biaffine_output_dir", type=str, required=True,
                        help="The output data directory from biaffine.")
    parser.add_argument("--token_offset_mapping_file", type=str,
                        help="The output data directory.")
    parser.add_argument("--formatted_output_dir", type=str, required=True,
                        help="The final output data directory.")
    parser.add_argument("--do_format", default=0, type=int,
                        help="0=bio (not format change will be applied); 1=brat; 2=bioc")
    parser.add_argument("--do_copy", action='store_true',
                        help="if copy the original plain text to output folder")
    global_args = parser.parse_args()

    try:
        main(global_args)
    except Exception as ex:
        traceback.print_exc()