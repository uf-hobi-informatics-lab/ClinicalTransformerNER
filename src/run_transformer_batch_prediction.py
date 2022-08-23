# -*- coding: utf-8 -*-

"""
The input files must have offset information. In input file, for each word in line, it must have at least text, start, end, tag information
output file suffix will be set to .bio.txt
"""

import argparse
import os, sys, copy
import traceback
from pathlib import Path
from collections import defaultdict
import torch
import torch.multiprocessing as mp
import transformers
from packaging import version

sys.path.append(str(Path( __file__ ).parent.absolute()))

from common_utils.common_io import json_load, output_bio
from common_utils.common_log import LOG_LVLs
from common_utils.output_format_converter import main as format_converter
from transformer_ner.data_utils import (TransformerNerDataProcessor,
                                        transformer_convert_data_to_features)
from transformer_ner.task import (MODEL_CLASSES, _output_bio, load_model,
                                  predict)
from transformer_ner.transfomer_log import TransformerNERLogger

pytorch_version = version.parse(transformers.__version__)
assert pytorch_version >= version.parse('3.0.0'), \
    'we now only support transformers version >=3.0.0, but your version is {}'.format(pytorch_version)

def check_sample(idx, args):
    if hasattr(args,"process_idx") and hasattr(args,"gpu_nodes"):
        return (idx%len(args.gpu_nodes)) == args.process_idx
    else:
        return True

def add_subdir_to_path(p,subdir):
    _p = Path(p)
    if subdir is not None:
        print(str(_p.parent / subdir / _p.name))
        return str(_p.parent / subdir / _p.name)
    else:
        print(str(_p))
        return str(_p)
    
def main(args):
    label2idx = json_load(os.path.join(args.pretrained_model, "label2idx.json"))
    idx2label = {v: k for k, v in label2idx.items()}
    args.label2idx = label2idx
    args.idx2label = idx2label

    # get config, model and tokenizer
    model_config, _, model_tokenizer = MODEL_CLASSES[args.model_type]
    tokenizer = model_tokenizer.from_pretrained(args.pretrained_model, do_lower_case=args.do_lower_case)
    args.tokenizer = tokenizer

    if hasattr(args,"model"):
        model = args.model
    else:
        config = model_config.from_pretrained(args.pretrained_model, do_lower_case=args.do_lower_case)
        args.config = config
        args.use_crf = config.use_crf

        model = load_model(args, args.pretrained_model)
        model.to(args.device)

    ner_data_processor = TransformerNerDataProcessor()
    ner_data_processor.set_logger(args.logger)
    if args.data_has_offset_information:
        ner_data_processor.offset_info_available()

    for subdir in args.subdirs:
        _args = copy.deepcopy(args)
        _args.preprocessed_text_dir = add_subdir_to_path(_args.preprocessed_text_dir,subdir)
        _args.output_dir_brat       = add_subdir_to_path(_args.output_dir_brat,subdir)
        _args.output_dir            = add_subdir_to_path(_args.output_dir,subdir)
        _args.raw_text_dir          = add_subdir_to_path(_args.raw_text_dir,subdir)

        ner_data_processor.set_data_dir(_args.preprocessed_text_dir)
        # fids = [each.stem.split(".")[0] for each in Path(args.preprocessed_text_dir).glob("*.txt")]
        labeled_bio_tup_lst = defaultdict(dict)
        for i, each_file in enumerate(Path(_args.preprocessed_text_dir).glob("*.txt")):
            if not check_sample(i, _args):
                continue
            try:
                test_example = ner_data_processor.get_test_examples(file_name=each_file.name, use_bio=_args.use_bio) #[(nsent, offsets, labels)]
                test_features = transformer_convert_data_to_features(args=_args,
                                                                        input_examples=test_example,
                                                                        label2idx=label2idx,
                                                                        tokenizer=tokenizer,
                                                                        max_seq_len=_args.max_seq_length)
                predictions = predict(_args, model, test_features)
                
                if _args.use_bio:
                    Path(_args.output_dir).mkdir(parents=True, exist_ok=True)
                    ofn = each_file.stem.split(".")[0] + ".bio.txt"
                    _args.predict_output_file = os.path.join(_args.output_dir, ofn)
                    _output_bio(_args, test_example, predictions)
                else:
                    labeled_bio_tup_lst[each_file.name]['sents'] = _output_bio(_args, test_example, predictions, save_bio=False)
                    with open(each_file, "r") as f:
                        labeled_bio_tup_lst[each_file.name]['raw_text'] = f.read()
            except Exception as ex:
                args.logger.error(f"Encountered an error when processing predictions for file: {each_file.name}")
                args.logger.error(traceback.format_exc())

        if _args.do_format:
            output_formatted_dir = Path(_args.output_dir_brat) if _args.output_dir_brat else Path(_args.output_dir).parent / "{}_formatted_output".format(Path(_args.output_dir).stem)  
            output_formatted_dir.mkdir(parents=True, exist_ok=True)
            format_converter(text_dir=_args.raw_text_dir,
                            input_bio_dir=(_args.output_dir if _args.use_bio else _args.raw_text_dir),
                            output_dir=output_formatted_dir,
                            formatter=_args.do_format,
                            do_copy_text=_args.do_copy,
                            labeled_bio_tup_lst=labeled_bio_tup_lst,
                            use_bio=_args.use_bio)

def argparser(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", default='bert', type=str, required=True,
                        help="valid values: bert, roberta or xlnet, albert, distilbert")
    parser.add_argument("--pretrained_model", type=str, required=True,
                        help="The pretrained model file or directory for fine tuning.")
    parser.add_argument("--preprocessed_text_dir", type=str, required=True,
                        help="The input data directory (bio with (dummy) label).")
    parser.add_argument("--raw_text_dir", type=str, required=True,
                        help="The input data directory (encoded text).")
    parser.add_argument("--data_has_offset_information", action='store_true',
                        help="Whether data has offset information.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output data directory (labeled bio).")
    parser.add_argument("--output_dir_brat", type=str,  default=None,
                        help="The output data directory (brat). Default: output_dir.parent / 'output_dir.stem'_formatted_output")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="maximum number of tokens allowed in each sentence")
    parser.add_argument("--log_file", default=None,
                        help="where to save the log information")
    parser.add_argument("--log_lvl", default="i", type=str,
                        help="d=DEBUG; i=INFO; w=WARNING; e=ERROR")
    parser.add_argument("--do_format", default=0, type=int,
                        help="0=bio (not format change will be applied); 1=brat; 2=bioc")
    parser.add_argument("--do_copy", action='store_true',
                        help="if copy the original plain text to output folder")
    parser.add_argument("--progress_bar", action='store_true',
                        help="show progress during the training in tqdm")
    parser.add_argument("--no_bio", action='store_true', default=False,
                        help="whether to use orignial text as input")
    parser.add_argument("--gpu_nodes", nargs="+", default=None,
                        help="use multiple gpu nodes")
    
    if args is None:
        parsed_args = parser.parse_args()
    else:
        parsed_args = parser.parse_args(args)
        
    parsed_args.use_bio = not parsed_args.no_bio
    return parsed_args

def multiprocessing_wrapper(args):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_nodes))
    
    print("Use GPU devices: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    
    N_gpus = len(args.gpu_nodes)
    args.N_gpus = N_gpus
    args_lst = []
    for i in range(N_gpus):
        _args = copy.deepcopy(args)
        _args.logger = TransformerNERLogger(_args.log_file, _args.log_lvl).get_logger()
        
        model_config, _, _ = MODEL_CLASSES[_args.model_type]
        config = model_config.from_pretrained(_args.pretrained_model, do_lower_case=_args.do_lower_case)
        _args.config = config
        _args.use_crf = config.use_crf
        _args.model = load_model(_args, _args.pretrained_model)
        _args.device = torch.device("cuda",i)
        _args.model.to(_args.device)
        _args.process_idx = i
        args_lst.append(_args)

    with mp.Pool(N_gpus) as p:
        p.map(main, args_lst)

if __name__ == '__main__':
    global_args = argparser()
    
    if global_args.gpu_nodes is None:
        # create logger
        logger = TransformerNERLogger(global_args.log_file, global_args.log_lvl).get_logger()
        global_args.logger = logger
        # device
        global_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Task will use cuda device: GPU_{}.".format(torch.cuda.current_device())
                    if torch.cuda.device_count() else 'Task will use CPU.')

        main(global_args)
    else:
        multiprocessing_wrapper(global_args)