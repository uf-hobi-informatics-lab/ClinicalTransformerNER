# Clinical Transformer NER

## Aim
The package is the implementation of a transformer based NER system for clinical information extraction task. We aim to provide a simple and quick tool for researchers to conduct clinical NER without comprehensive knowledge of transformers. We also handle the sequence with length longer than the general transformer limits (512 tokens).

## Current available models
- BERT
- RoBERTa
- ALBERT
- ELECTRA
- DistilBERT
- XLNet

## Usage and example
- Training and test with BIO 

```shell script
# set GPU
export CUDA_VISIBLE_DEVICES=0

# use bert
python src/run_transformer_ner.py \
      --model_type bert \
      --pretrained_model bert-base-uncased \
      --data_dir ./test_data/conll-2003 \
      --new_model_dir ./new_bert_ner_model \
      --overwrite_model_dir \
      --predict_output_file ./bert_pred.txt \
      --max_seq_length 256 \
      --save_model_core \
      --do_train \
      --do_predict \
      --model_selection_scoring strict-f_score-1 \
      --do_lower_case \
      --train_batch_size 8 \
      --eval_batch_size 8 \
      --train_steps 500 \
      --learning_rate 1e-5 \
      --num_train_epochs 1 \
      --gradient_accumulation_steps 1 \
      --do_warmup \
      --seed 13 \
      --warmup_ratio 0.1 \
      --max_num_checkpoints 3 \
      --log_file ./log.txt \
      --progress_bar \
      --early_stop 3
```

- Test on multiple files and convert bio to brat format

```shell script
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
```

## Organization
- Department of Health Outcomes and Biomedical Informatics, College of Medicine, University of Florida

## authors
- Xi Yang
- Jiang Bian
- Yonghui Wu

## contact
raise issue in our repo

## reference
please cite our paper:


## MIMIC-III pre-trained models
- https://transformer-models.s3.amazonaws.com/mimiciii_albert_10e_128b.zip
- https://transformer-models.s3.amazonaws.com/mimiciii_bert_10e_128b.zip
- https://transformer-models.s3.amazonaws.com/mimiciii_electra_5e_128b.zip
- https://transformer-models.s3.amazonaws.com/mimiciii_roberta_10e_128b.zip

