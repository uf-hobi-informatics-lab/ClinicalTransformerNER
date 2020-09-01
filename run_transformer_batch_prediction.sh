: '
The script is used to run multi-file batch prediction using transformer ner
We only use bert as example, the roberta, XLNet should be the same
The input files must have offset information
If no offset information, just combine all the files into one test.txt and use the do_pred from run_transformer_ner.sh for prediction
This script is design for mainly production using to generate brat/BioC formatted outputs with offset information.
'

################# BERT example #####################
export CUDA_VISIBLE_DEVICES=0

# config and tokenizer information can be found in the pretrained model dir
# use format 1 for BRAT, 2 for BioC, 0 as default for BIO
python ./src/run_transformer_batch_prediction.py \
      --model_type bert \
      --pretrained_model <your pretrained model path> \
      --raw_text_dir <path to the original text files> \
      --preprocessed_text_dir <path to the bio formatted files> \
      --output_dir <path to save predicted results> \
      --max_seq_length 128 \
      --do_lower_case \
      --eval_batch_size 8 \
      --log_file ./log.txt\
      --do_format 1 \
      --do_copy \
      --data_has_offset_information