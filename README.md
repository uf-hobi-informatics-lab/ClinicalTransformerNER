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
- Xi Yang (alexgre@ufl.edu)
- Jiang Bian
- Yonghui Wu

## contact
raise issue in our repo; or contact alexgre@ufl.edu

## reference
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

