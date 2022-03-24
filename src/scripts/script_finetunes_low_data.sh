export CUDA_VISIBLE_DEVICES=0

declare -a DATASET=("xsum")
EXP_FOLDER=finetunes_low_data/full_model
START_GROUP=1
END_GROUP=1

###### Training with 10 examples #####
for dataset in ${DATASET[@]}
do
	for g in $(seq $START_GROUP $END_GROUP)
	do
		python main.py \
			--dataset_name "$dataset" \
			--model_name_or_path facebook/bart-base \
			--output_dir ../results/"$EXP_FOLDER"/"$dataset"_10_examples/group_"$g" \
			--logging_dir ../results/"$EXP_FOLDER"/"$dataset"_10_examples/group_"$g"/logs/ \
			--report_to tensorboard \
			--overwrite_output_dir \
			--evaluation_strategy steps \
			--save_strategy steps \
			--max_steps 100 \
			--eval_steps 2 \
			--save_steps 2 \
			--save_total_limit 1 \
			--logging_steps 1 \
			--per_device_train_batch_size 2 \
			--per_device_eval_batch_size 16 \
			--lr_scheduler_type polynomial \
			--learning_rate 3e-5 \
			--do_train \
			--do_eval \
			--gradient_accumulation_steps 4 \
			--max_train_samples 10 \
			--max_val_samples 1000 \
			--label_smoothing_factor 0.1 \
			--weight_decay 0.01 \
			--max_grad_norm 0.1 \
			--warmup_steps 25 \
			--select_start_indice "$(( ($g-1)*10 ))" \
			--predict_with_generate \
			--save_model_accord_to_rouge
	done
done

##### Training with 100 examples #####
#for dataset in ${DATASET[@]}
#do
#	for g in $(seq $START_GROUP $END_GROUP)
#	do
#		python main.py \
#			--dataset_name "$dataset" \
#			--model_name_or_path facebook/bart-base \
#			--output_dir ../results/"$EXP_FOLDER"/"$dataset"_100_examples/group_"$g" \
#			--logging_dir ../results/"$EXP_FOLDER"/"$dataset"_100_examples/group_"$g"/logs/ \
#			--report_to tensorboard \
#			--overwrite_output_dir \
#			--evaluation_strategy steps \
#			--save_strategy steps \
#			--max_steps 200 \
#			--eval_steps 5 \
#			--save_steps 5 \
#			--save_total_limit 1 \
#			--logging_steps 1 \
#			--per_device_train_batch_size 2 \
#			--per_device_eval_batch_size 16 \
#			--lr_scheduler_type polynomial \
#			--learning_rate 3e-5 \
#			--do_train \
#			--do_eval \
#			--gradient_accumulation_steps 16 \
#			--max_train_samples 100 \
#			--max_val_samples 1000 \
#			--label_smoothing_factor 0.1 \
#			--weight_decay 0.01 \
#			--max_grad_norm 0.1 \
#			--warmup_steps 20 \
#			--select_start_indice "$(( ($g-1)*100 ))" 
#	done
#done

####### Testing with 10 examples #####
#for dataset in ${DATASET[@]}
#do
#	if [ "$dataset" = "cnn_dailymail" ]
#	then
#		CKPT=("1" "2" "3" "4" "5")
#	elif [ "$dataset" = "xsum" ]
#	then
#		CKPT=("6" "7" "8" "9" "10")
#	fi
#
#	for g in $(seq $START_GROUP $END_GROUP)
#	do
#		#echo ${CKPT[$(($g-1))]}
#		python main.py \
#			--dataset_name "$dataset" \
#			--model_name_or_path ../results/"$EXP_FOLDER"/"$dataset"_10_examples/group_"$g"/checkpoint-"${CKPT[$(($g-1))]}" \
#			--output_dir ../results/"$EXP_FOLDER"/"$dataset"_10_examples/group_"$g"/checkpoint-"${CKPT[$(($g-1))]}" \
#			--logging_dir ../results/"$EXP_FOLDER"/"$dataset"_10_examples/group_"$g"/checkpoint-"${CKPT[$(($g-1))]}"/logs \
#			--report_to tensorboard \
#			--overwrite_output_dir \
#			--per_device_eval_batch_size 48 \
#			--do_predict \
#			--predict_with_generate
#	done
#done

####### Testing with 100 examples #####
#for dataset in ${DATASET[@]}
#do
#	if [ "$dataset" = "cnn_dailymail" ]
#	then
#		CKPT=("1" "2" "3" "4" "5")
#	elif [ "$dataset" = "xsum" ]
#	then
#		CKPT=("6" "7" "8" "9" "10")
#	fi
#
#	for g in $(seq $START_GROUP $END_GROUP)
#	do
#		#echo ${CKPT[$(($g-1))]}
#		python main.py \
#			--dataset_name "$dataset" \
#			--model_name_or_path ../results/"$EXP_FOLDER"/"$dataset"_100_examples/group_"$g"/checkpoint-"${CKPT[$(($g-1))]}" \
#			--output_dir ../results/"$EXP_FOLDER"/"$dataset"_100_examples/group_"$g"/checkpoint-"${CKPT[$(($g-1))]}" \
#			--logging_dir ../results/"$EXP_FOLDER"/"$dataset"_100_examples/group_"$g"/checkpoint-"${CKPT[$(($g-1))]}"/logs \
#			--report_to tensorboard \
#			--overwrite_output_dir \
#			--per_device_eval_batch_size 48 \
#			--do_predict \
#			--predict_with_generate
#	done
#done

