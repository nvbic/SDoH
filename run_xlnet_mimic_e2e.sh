export CUDA_VISIBLE_DEVICES=0
bz=4
epn=3
sc=2
dfmm=0
model_type=xlnet
pm=../pretrained_model/mimiciii_xlnet_5e_128b
data_dir=../data/sdoh_relation_e2e_aio_th1
nmd=/home/zehao.yu/workspace/py3/SDoH/model/relations_model/xlnet_mimic
pof=/home/zehao.yu/workspace/py3/SDoH/res/predictions_xlnet_mimic_large_e2e.txt
log=/home/zehao.yu/workspace/py3/SDoH/logs/log_relation_xlnet_mimic.txt

export CUDA_VISIBLE_DEVICES=1
python3 /home/zehao.yu/workspace/py3/dr_ann/ClinicalTransformerRelationExtraction/src/relation_extraction.py \
                --model_type $model_type \
                --data_format_mode $dfmm \
                --classification_scheme $sc \
                --pretrained_model $pm \
                --data_dir $data_dir \
                --new_model_dir $nmd \
                --predict_output_file $pof \
                --overwrite_model_dir \
                --seed 13 \
                --max_seq_length 512 \
                --cache_data \
                --do_predict \
                --do_lower_case \
                --train_batch_size $bz \
                --eval_batch_size $bz \
                --learning_rate 1e-5 \
                --num_train_epochs $epn \
                --gradient_accumulation_steps 1 \
                --do_warmup \
                --warmup_ratio 0.1 \
                --weight_decay 0 \
                --max_num_checkpoints 0 \
                --log_file $log 
edr=/home/zehao.yu/workspace/py3/SDoH/res/bert_final_formatted_output
pod=/home/zehao.yu/workspace/py3/SDoH/res/relation_predicted_results_xlnet_mimic_e2e
python3 /home/zehao.yu/workspace/py3/dr_ann/ClinicalTransformerRelationExtraction/src/data_processing/post_processing.py \
                --mode mul \
                --predict_result_file $pof \
                --entity_data_dir $edr \
                --test_data_file ${data_dir}/test.tsv \
                --brat_result_output_dir $pod\
                --log_file $log