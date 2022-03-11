# Clinical Transformer NER

## Aim
The package is the implementation of a transformer based NER system for clinical information extraction task. We aim to provide a simple and quick tool for researchers to conduct clinical NER without comprehensive knowledge of transformers. We also implemented a strategy to handle the sequence with length longer than the general transformer limits (512 tokens) without truncating any tokens.

## Current available models
- BERT (base, large, mimiciii-pretrained)
- RoBERTa (base, large, mimiciii-pretrained)
- ALBERT (base, large, xlarge, xxlarge, mimiciii-pretrained)
- ELECTRA (base, large, mimiciii-pretrained)
- DistilBERT (base)
- XLNet (base, large, mimiciii-pretrained)
- Longformer (allenai/longformer-base-4096, allenai/longformer-large-4096)
- DeBERTa (microsoft/deberta-base, microsoft/deberta-large, microsoft/deberta-xlarge)
> note: 1. all mimic-pretrained models are based on base transformer architecture (Download is available in the section MIMIC-III pre-trained models); 2. DeBERTa is not support xlarge-v2 due to tokenizer change in original implementation

## Usage and example (sequence labeling approach)
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
##### note ######
# In the script below, you are asked to provide a preprocessed_text_dir which contains all the preprocessed file.
# 
# If you only use the BIO format for output (you have to remove --data_has_offset_information flag
# and set --do_format flag to 0), and the data format will be the format exactly as the conll-2003 dataset.
# 
# If you need BRAT or BioC format as output (as the example script), then you have to add offset information 
# to the BIO data to indicate where each word is located in the raw text. 
# We suggest you to follow the format below:
# 
# The original sentences: "Name: John Doe\nAge: 18"
# The two sentences after preprocesing "Name : John Doe\nAge : 18"
# 
# then, you can convert the data into BIO format similar as the Conll-2003 as
# """
# Name 0 4 0 4 O
# : 4 5 5 6 O
# John 6 10 7 11 B-name
# Doe 11 14 12 15 I-name
# 
# Age 15 18 16 19 O
# : 18 19 19 20 O
# 18 20 22 22 24 B-age
# 
# For test purposes, you do not need to assign a real BIO label for each word, 
# you can just simple assign "O" to all of them. 
# It will not influence the prediction results since the predictions will be converted to brat/BioC, 
# and you need to use those for evaluation.
# """
# 
# The first two numbers are the offsets of a word in the original text and the following 
# two numbers are the offsets of a word in the preprocessed text. 
# If you do not need to perform any preprocessing, then you have to set the second set of offsets as the first one.
#################

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

####
# note: If you use do_format, then we have two outputs: 
# 1) all bio outputs in output_dir; 
# 2) 2) we create a formatted output dir (this directory's name is output_dir's name with a suffix of '_formatted_output') for the formatted # outputs (brat format if you set do_format=1). If you set --do_copy, we will copy the .txt files to the formatted output dir, otherwise we only put .ann files in the formatted output dir.
####
```

## Usage and example (biaffine approach)
- implementation of https://aclanthology.org/2020.acl-main.577.pdf
- see tutorial/convert_other_format_data_to_biaffine_format.ipynb for how to construct data from brat or bio format for biaffine model
- training with biaffine, you just need to set the --use_biaffine flag
- you can use any transformers as encoder, we use biaffine to decode
- you cannot use biaffine with CRF; and you cannot save core model (transformers)
- options --mlp_dim (default 128) and --mlp_layers (default 0) (see wiki for more details)
- default biaffine prediction results will be in json format
- you can use run_format_biaffine_output.py to convert format to BIO or brat
- example
```shell
# training and prediction (predict to biaffine format)
export CUDA_VISIBLE_DEVICES=0
python src/run_transformer_ner.py \
      --use_biaffine \
      --mlp_dim 128 \
      --mlp_layers 0 \
      --model_type bert \
      --pretrained_model bert-base-uncased \
      --data_dir ./test_data/biaffine_conll2003 \
      --new_model_dir ./new_bert_biaffine_ner_model \
      --overwrite_model_dir \
      --predict_output_file ./bert_biaffine_pred.json \
      --max_seq_length 512 \
      --do_train \
      --do_predict \
      --do_lower_case \
      --train_batch_size 4 \
      --eval_batch_size 32 \
      --train_steps 1000 \
      --learning_rate 5e-5 \
      --min_lr 5e-6 \
      --num_train_epochs 50 \
      --gradient_accumulation_steps 1 \
      --do_warmup \
      --warmup_ratio 0.1 \
      --seed 13 \
      --max_num_checkpoints 1 \
      --log_file ./log.txt \
      --progress_bar \
      --early_stop 5
```

- reformat output to brat and do evaluation
```shell
# to BIO format
python run_format_biaffine_output.py \
  --raw_input_dir_or_file <where the test BIO data located> \
  --biaffine_output_file <where is the biaffine output json file> \
  --formatted_output_dir <where is the formatted output, we will create a predict.txt under this folder>

# BIO evaluation
python eval_scripts/new_bio_eval.py -f1 ./test_data/conll-2003/test.txt -f2 <formatted output file>

# To Brat format
python run_format_biaffine_output.py \
  --raw_input_dir_or_file <where the test BIO data located> \
  --biaffine_output_file <where is the biaffine output json file> \
  --mapping_file <a pickle file generated based on test data for mapping file id and offsets> \
  --do_copy_raw_text True \
  --formatted_output_dir <where is the formatted output, we create all .ann under this folder>
  
# brat evaluation
python eval_scripts/brat_eval.py --f1 <gold standard ann files dir> --f2 <formatted output dir>
```

## Tutorial
> we have tutorials in the tutorial directory
- brat2bio.ipynb is an example on how to write code to covert brat format annotation to BIO format which used for training and test 
- pipeline_preprocessing_training_prediction.ipynb is a full pipeline example from data preprocessing to training to prediction to evaluation
- convert_other_format_data_to_biaffine_format.ipynb is for how to generate data for biaffine model from other formats (BIO, BRAT)
- note: in full pipeline example, we used our NLPreprocessing package which is customized for clinical notes but may have issues if the notes have some unique format in it.

## Wiki for all parameters
[wiki link to description of all arguments](https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER/wiki/Parameters)

## Organization
- Department of Health Outcomes and Biomedical Informatics, College of Medicine, University of Florida

## Authors
- Yonghui Wu (yonghui.wu@ufl.edu) (main contact)
- Jiang Bian (bianjiang@ufl.edu)
- Xi Yang (alexgre@ufl.edu)

## Contact
- If you have any questions, please raise an issue in the GitHub

## Reference
please cite our paper:
> Xi Yang, Jiang Bian, William R Hogan, Yonghui Wu, Clinical concept extraction using transformers, Journal of the American Medical Informatics Association, ocaa189, https://doi.org/10.1093/jamia/ocaa189

```
@article{10.1093/jamia/ocaa189,
    author = {Yang, Xi and Bian, Jiang and Hogan, William R and Wu, Yonghui},
    title = "{Clinical concept extraction using transformers}",
    journal = {Journal of the American Medical Informatics Association},
    year = {2020},
    month = {10},
    abstract = "{The goal of this study is to explore transformer-based models (eg, Bidirectional Encoder Representations from Transformers [BERT]) for clinical concept extraction and develop an open-source package with pretrained clinical models to facilitate concept extraction and other downstream natural language processing (NLP) tasks in the medical domain.We systematically explored 4 widely used transformer-based architectures, including BERT, RoBERTa, ALBERT, and ELECTRA, for extracting various types of clinical concepts using 3 public datasets from the 2010 and 2012 i2b2 challenges and the 2018 n2c2 challenge. We examined general transformer models pretrained using general English corpora as well as clinical transformer models pretrained using a clinical corpus and compared them with a long short-term memory conditional random fields (LSTM-CRFs) mode as a baseline. Furthermore, we integrated the 4 clinical transformer-based models into an open-source package.The RoBERTa-MIMIC model achieved state-of-the-art performance on 3 public clinical concept extraction datasets with F1-scores of 0.8994, 0.8053, and 0.8907, respectively. Compared to the baseline LSTM-CRFs model, RoBERTa-MIMIC remarkably improved the F1-score by approximately 4\\% and 6\\% on the 2010 and 2012 i2b2 datasets. This study demonstrated the efficiency of transformer-based models for clinical concept extraction. Our methods and systems can be applied to other clinical tasks. The clinical transformer package with 4 pretrained clinical models is publicly available at https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER. We believe this package will improve current practice on clinical concept extraction and other tasks in the medical domain.}",
    issn = {1527-974X},
    doi = {10.1093/jamia/ocaa189},
    url = {https://doi.org/10.1093/jamia/ocaa189},
    note = {ocaa189},
    eprint = {https://academic.oup.com/jamia/advance-article-pdf/doi/10.1093/jamia/ocaa189/34055422/ocaa189.pdf},
}
```

## MIMIC-III pre-trained models
- https://transformer-models.s3.amazonaws.com/mimiciii_albert_10e_128b.zip
- https://transformer-models.s3.amazonaws.com/mimiciii_bert_10e_128b.zip
- https://transformer-models.s3.amazonaws.com/mimiciii_electra_5e_128b.zip
- https://transformer-models.s3.amazonaws.com/mimiciii_roberta_10e_128b.zip
- https://transformer-models.s3.amazonaws.com/mimiciii_xlnet_5e_128b.zip
- https://transformer-models.s3.amazonaws.com/mimiciii_deberta_10e_128b.tar.gz
- https://transformer-models.s3.amazonaws.com/mimiciii_longformer_5e_128b.zip
> note: all model pretraining tasks were done with the scripts at https://github.com/huggingface/transformers/tree/master/examples/language-modeling with a few customization.
