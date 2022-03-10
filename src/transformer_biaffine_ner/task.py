# coding=utf-8
# Created by bugface (https://github.com/bugface)
# First created at 12/31/21

from pathlib import Path
import gc

from common_utils.common_io import json_dump, json_load
from transformer_biaffine_ner.task_utils import train, predict, get_tokenizer, get_config
from transformer_biaffine_ner.data_utils import TransformerNerBiaffineDataProcessor
from transformer_ner.task import set_seed
from transformers import AutoModel


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
    new_model_dir_path.mkdir(parents=True, exist_ok=True)

    # init data preprocessor
    data_processor = TransformerNerBiaffineDataProcessor()
    data_processor.set_data_dir(args.data_dir)
    data_processor.set_logger(args.logger)
    data_processor.set_max_seq_len(args.max_seq_length)
    data_processor.set_tokenizer_type(args.model_type)
    # default to no cache, since it will take many disk space
    # we use parallel to do preprocessing which is fast enough to avoid caching
    # in future we need a flag to control this
    data_processor.set_cache(False)

    if args.do_train:
        # create tokenizer and add it to data processor
        args.tokenizer = get_tokenizer(args, is_train=True)
        data_processor.set_tokenizer(args.tokenizer)

        # create config; we need to get the num_labels before get config
        label2idx, _ = data_processor.get_labels()
        args.label2idx = label2idx
        args.num_classes, args.idx2label = _get_unique_num_classes(label2idx)
        args.config = get_config(args, is_train=True)
        args.config.num_labels = args.num_classes
        args.config.label2idx = args.label2idx
        args.config.idx2label = args.idx2label
        args.config.vocab_size = len(args.tokenizer)
        args.config.mlp_dim = args.mlp_dim
        args.config.mlp_layers = args.mlp_layers
        args.config.mlp_hidden_dim = args.mlp_hidden_dim
        args.config.use_focal_loss = args.focal_loss
        args.config.focal_loss_gamma = args.focal_loss_gamma
        args.logger.info(f"load pretrained model from {args.pretrained_model}")
        if args.config.use_focal_loss:
            args.logger.info(f"Using focal loss with gamma = {args.config.focal_loss_gamma}")
        else:
            args.logger.info(f"Using cross entropy loss")

        #############################################
        # also influence TransformerBiaffineNerModel AutoModel load
        # we need to save pretrained_model to new_model_dir
        # in this way we can init transformer when we do prediction
        # args.config.base_model_path = args.pretrained_model
        # AutoModel.from_pretrained(args.config.base_model_path).save_pretrained(args.new_model_dir)
        # args.config.base_model_path = args.new_model_dir
        # # very bad design above
        # use AutoModel.from_config to random init so we do not need to save original model for load model
        # add flag in config to control behavior train - load from pretrained model; pred - init random
        args.config.base_model_path = args.pretrained_model
        args.config.init_in_training = True
        #############################################

        # get train, dev data loader
        train_data_loader = data_processor.get_train_data_loader()
        dev_data_loader = data_processor.get_dev_data_loader()

        train(args, train_data_loader, dev_data_loader)

        # if do_train and do_predict, we try to release some RAM here
        # del train_data_loader
        # del dev_data_loader
        gc.collect()

    if args.do_predict:
        args.tokenizer = get_tokenizer(args, is_train=False)
        data_processor.set_tokenizer(args.tokenizer)
        data_processor.set_label2idx(json_load(Path(args.new_model_dir) / "label2idx.json"))
        test_data_loader = data_processor.get_test_data_loader()

        args.config = get_config(args, is_train=False)
        args.config.init_in_training = False
        args.config.idx2label = {v: k for k, v in args.config.label2idx.items()}
        args.logger.info(f"configuration for prediction:\n{args.config}")

        # predict_results format: [{"tokens": [xx ...], "entities": [(en, en_type, s, e) ...]}]
        # note: we add 1 to the end position e so you can use e directly in list slice
        predicted_results = predict(args, test_data_loader)

        # we just output json formatted results
        # we let users to do reformat using run_format_biaffine_output.py
        output_fn = args.predict_output_file if args.predict_output_file else Path(args.new_model_dir) / "predict.json"
        json_dump(predicted_results, output_fn)


def _get_unique_num_classes(label2idx):
    idx2label = {v: k for k, v in label2idx.items() if v != 0}
    # we need to include 0
    num_classes = len(idx2label) + 1

    return num_classes, idx2label
