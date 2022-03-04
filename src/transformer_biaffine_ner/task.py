# coding=utf-8
# Created by bugface (https://github.com/bugface)
# First created at 12/31/21

from pathlib import Path

from common_utils.common_io import json_dump
from transformer_biaffine_ner.task_utils import train, predict, get_tokenizer, get_config
from transformer_biaffine_ner.data_utils import TransformerNerBiaffineDataProcessor
from transformer_ner.task import set_seed


def run_task(args):
    args.logger.info("Training with Biaffine mode...")

    if args.resume_from_model:
        raise NotImplementedError("we currently not support resume training for biaffine mode.")
    if args.save_model_core:
        raise NotImplementedError("we currently not support save model core for biaffine mode.")
    if args.adversarial_training_method:
        raise NotImplementedError("we currently not support adversarial training for biaffine mode.")

    set_seed(args.seed)

    new_model_dir_path = Path(args.new_model_dir)
    if new_model_dir_path.exists() and len(list(new_model_dir_path.glob("*.bin"))) > 0 \
            and args.do_train and not args.overwrite_model_dir:
        raise ValueError(
            """new model directory: {} exists. 
            Use --overwrite_model_dir to overwrite the previous model. 
            Or create another directory for the new model""".format(args.new_model_dir))

    # init data preprocessor
    data_processor = TransformerNerBiaffineDataProcessor()
    data_processor.set_data_dir(args.data_dir)
    data_processor.set_logger(args.logger)
    data_processor.set_max_seq_len(args.max_seq_length)

    if args.do_train:
        # create tokenizer and add it to data processor
        args.tokenizer = get_tokenizer(args, is_train=True)
        data_processor.set_tokenizer(args.tokenizer)
        # create config; we need to get the num_labels before get config
        label2idx, _ = data_processor.get_labels()
        args.label2idx = label2idx
        args.num_classes, args.idx2label = _get_unique_num_classes
        args.config.num_labels = args.num_classes
        args.config = get_config(args, is_train=True)
        args.config.vocab_size = len(args.tokenizer)
        args.config.mlp_dim = args.mlp_dim
        args.config.mlp_layers = args.mlp_layers

        # get train, dev data loader
        train_data_loader = data_processor.get_train_data_loader()
        dev_data_loader = data_processor.get_dev_data_loader()

        train(args, train_data_loader, dev_data_loader)

        args.tokenizer.save_pretrained(args.new_model_dir)
        args.config.save_pretrained(args.new_model_dir)

    if args.do_predict:
        args.tokenizer = get_tokenizer(args, is_train=False)
        data_processor.set_tokenizer(args.tokenizer)
        test_data_loader = data_processor.get_test_data_loader()

        args.config = get_config(args, is_train=False)
        # predict_results format: [{"tokens": [xx ...], "entities": [(en, en_type, s, e) ...]}]
        predicted_results = predict(args, test_data_loader)

        # we just output json formatted results
        # we let users to do reformat using run_format_biaffine_output.py
        output_fn = args.predict_output_file if args.predict_output_file else Path(args.new_model_dir) / "predicts.json"
        json_dump(predicted_results, output_fn)


def _get_unique_num_classes(label2idx):
    idx2label = {v: k for k, v in label2idx if v != 0}
    num_classes = len(idx2label)

    return num_classes, idx2label
