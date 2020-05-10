: '
The script is used to run multi-file batch prediction using transformer ner
We only use bert as example, the roberta, XLNet should be the same
The input files must have offset information
If no offset information, just combine all the files into one test.txt and use the do_pred from run_transformer_ner.sh for prediction
This script is design for mainly production using.
'

################# BERT example #####################
export CUDA_VISIBLE_DEVICES=0

# config and tokenizer information can be found in the pretrained model dir
# use format 1 for BRAT, 2 for BioC, 0 as default for BIO
python src/run_transformer_ner.py \
      --model_type bert \
      --pretrained_model ./new_bert_ner_model \
      --data_dir ./test_data/conll-2003 \
      --output_dir ./bert_pred.txt \
      --max_seq_length 256 \
      --do_lower_case \
      --eval_batch_size 8 \
      --log_file ./log.txt\
      --do_format 1 \
      --do_copy \
      --data_has_offset_information
