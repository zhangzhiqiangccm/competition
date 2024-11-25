
i=1
model_path="/data_local/bert-models/Salesforce/SFR-Embedding-2_R"
model_version="zero_round"
lora_path="none"
CUDA_VISIBLE_DEVICES=$i nohup python3 -u recall.py ${model_path} ${model_version} ${lora_path} > log_recall_zero 2>&1


nohup sh run_mistral_cos_argu.sh > log_simcse 2>&1


model_version="v3_round1_qlora"
lora_path="./model_save/${model_version}_rerun/epoch_19_model/adapter.bin"
CUDA_VISIBLE_DEVICES=$i nohup python3 -u recall.py ${model_path} ${model_version} ${lora_path} > log_recall_train 2>&1
