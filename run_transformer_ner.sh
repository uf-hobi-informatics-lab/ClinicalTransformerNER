: '
The script contains the example shell commands you can use to run the transformer_ner tasks
We include two groups of commands:
1. train and predict: The commands in train and predict section demonstrated how to run training and prediction in sequence
2. only predict: If you have a trained model, you can run prediction only following the commands in the only predict section
3. the prediction here is only for one file (test.txt) prediction. If you need batch prediction on group of files, use run_transformer_batch_prediction.sh instead.

Each section contains an example for BERT
We do support ALBERT, DISTILBERT, XLNet, RoBERTa as well. You can find more model information at https://huggingface.co/transformers/pretrained_models.html.
We did not include examples using fp16 training mode but you can train model with fp16 (read run_transformer_ner.py source code)
We currently do not support distraibuted multi-GPU training since fine-tuning task is not heavy on most clinical NER datasets.
'

########################### train and predict ###########################
# tell system which GPU to use
export CUDA_VISIBLE_DEVICES=0

########################### train and predict ###########################
#bert
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