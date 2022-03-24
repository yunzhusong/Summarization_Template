export CUDA_VISIBLE_DEVICES=0

declare -a DATASET=("reddit")
EXP_FOLDER=dev
EVAL_DATASET="xsum"
START_GROUP=1
END_GROUP=1

for dataset in ${DATASET[@]}
do
	for g in $(seq $START_GROUP $END_GROUP)
	do
		python main.py \
			--dataset_name "$dataset" \
			--eval_dataset_name "$EVAL_DATASET" \
			--model_name_or_path facebook/bart-base \
			--output_dir ../results/"$EXP_FOLDER"/"$dataset"_eval_"$EVAL_DATASET" \
			--logging_dir ../results/"$EXP_FOLDER"/"$dataset"_eval_"$EVAL_DATASET"/logs/ \
			--report_to tensorboard \
			--overwrite_output_dir \
			--evaluation_strategy steps \
			--save_strategy steps \
			--max_steps 800 \
			--eval_steps 2 \
			--save_steps 2 \
			--save_total_limit 1 \
			--logging_steps 10 \
			--per_device_train_batch_size 2 \
			--per_device_eval_batch_size 16 \
			--lr_scheduler_type constant \
			--learning_rate 3e-5 \
			--do_train \
			--do_eval \
			--gradient_accumulation_steps 16 \
			--max_train_samples 100 \
			--max_val_samples 10 \
			--label_smoothing_factor 0.1 \
			--weight_decay 0.01 \
			--max_grad_norm 0.1 \
			--warmup_steps 0 \
			--predict_with_generate \
			--save_model_accord_to_rouge
	done
done

