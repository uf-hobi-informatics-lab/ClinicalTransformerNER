# !/usr/bin/python
# -*- coding: utf-8 -*-

"""
The script is used to format the BIO output to target output format like BRAT
The script is used to help all the prediction without format
"""

import argparse
import traceback
from pathlib import Path

from common_utils.output_format_converter import main as format_converter


def main(args):
    base_path = Path(args.bio_dir)
    output_formatted_dir = base_path.parent / f"{base_path.stem}_formatted_output"
    output_formatted_dir.mkdir(parents=True, exist_ok=True)
    format_converter(text_dir=args.raw_text_dir,
                     input_bio_dir=args.bio_dir,
                     output_dir=output_formatted_dir,
                     formatter=args.do_format,
                     do_copy_text=args.do_copy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_text_dir", type=str, required=True,
                        help="The input data directory.")
    parser.add_argument("--bio_dir", type=str, required=True,
                        help="The output data directory.")
    parser.add_argument("--do_format", default=0, type=int,
                        help="0=bio (not format change will be applied); 1=brat; 2=bioc")
    parser.add_argument("--do_copy", action='store_true',
                        help="if copy the original plain text to output folder")
    global_args = parser.parse_args()

    try:
        main(global_args)
    except Exception as ex:
        traceback.print_exc()