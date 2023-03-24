
DATA_NAME='arguana'

python -m torch.distributed.launch --nproc_per_node 4 -m remop.driver.mencode \
  --output_dir ./encoded \
  --model_type bert \
  --model_name_or_path Luyu/co-condenser-marco \
  --fp16 \
  --prefix \
  --per_device_eval_batch_size 128 \
  --encode_in_path ./data/$DATA_NAME/corpus.jsonl \
  --encoded_save_path ./encoded/$DATA_NAME.corpus \
  --attribute_prompt_dir ./experiment_stage2/trained_prompts \
  --general_prompt_path ./experiment_stage2/trained_prompts/general_state_dict.prompt \


python -m torch.distributed.launch --nproc_per_node 4 -m remop.driver.mencode \
  --output_dir ./encoded \
  --model_type bert \
  --model_name_or_path Luyu/co-condenser-marco \
  --fp16 \
  --prefix \
  --encode_is_qry \
  --per_device_eval_batch_size 128 \
  --encode_in_path ./data/$DATA_NAME/queries.jsonl \
  --encoded_save_path ./encoded/$DATA_NAME.query \
  --attribute_prompt_dir ./experiment_stage2/trained_prompts \
  --general_prompt_path ./experiment_stage2/trained_prompts/general_state_dict.prompt