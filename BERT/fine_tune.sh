export MAX_LENGTH=128
export BERT_MODEL=sumerianBERTo
export OUTPUT_DIR=sumerianBERTo-finetune
export BATCH_SIZE=128
export NUM_EPOCHS=3
export SAVE_STEPS=1000
export SEED=1

cat POS_corpus/train.txt POS_corpus/dev.txt POS_corpus/test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > POS_corpus/labels.txt
python3 run_ner.py \
--data_dir ./POS_corpus \
--labels ./POS_corpus/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--overwrite_output_dir \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict
